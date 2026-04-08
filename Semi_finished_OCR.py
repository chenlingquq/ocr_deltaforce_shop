import sys
import re
import time
import json
import ctypes
import threading
from queue import Queue, Empty
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
from mss import mss
from pynput.mouse import Button, Controller
from rapidocr_onnxruntime import RapidOCR

from PySide6.QtCore import Qt, QRect, QPoint, QThread, Signal, QObject
from PySide6.QtGui import QPainter, QColor, QPen, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QPlainTextEdit,
    QMessageBox, QDialog
)

# ---- 可选：提升 DPI 感知（减少高分屏坐标错位概率）----
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

CONFIG_PATH = "ocr_clicker_config.json"
TIME_RE = re.compile(r"(?:(\d+)\s*分)?\s*(\d+)\s*秒")


# -----------------------------
# 数据结构
# -----------------------------
@dataclass
class ClickConfig:
    # OCR频率控制（capture线程和主OCR循环的目标节奏）
    ocr_interval_sec: float = 0.15

    # 点击1（刷新）两次触发点（秒）
    trigger1a_sec: int = 25
    trigger1b_sec: int = 20

    # 点击2触发点（秒）
    trigger2_sec: int = 10

    # 点击2后进入下级菜单：内部倒计时基于 5 秒（可改），再做毫秒微调
    countdown_after_click2_sec: float = 5.0
    click3_offset_ms: int = 0  # 可正可负：+晚点，-提前

    # 预处理
    binarize: bool = True
    threshold: int = 160

    # 稳定策略：连续 N 次识别到“合理值”才更新
    stable_required: int = 2

    # 点击1后做校准
    enable_calibration: bool = True
    calibration_window_sec: float = 2.0

    # 调试：保存 OCR 裁剪图（定位 DPI/ROI 是否跑偏）
    debug_save_crop: bool = False

    # ROI 归一化保存（跨分辨率/缩放恢复）
    roi_time: Optional[Dict[str, float]] = None
    roi_click1: Optional[Dict[str, float]] = None
    roi_click2: Optional[Dict[str, float]] = None
    roi_click3: Optional[Dict[str, float]] = None

    # 手动屏幕参数
    use_manual_screen: bool = False
    manual_phys_w: int = 3840
    manual_phys_h: int = 2160
    manual_scale_percent: int = 200  # 200% => scale=2.0

    # ====== 新增：Anchor 相关（解决“连续使用滞后/anchor偶发未设置”）======
    anchor_alpha: float = 0.25          # Anchor EMA 平滑系数（0.05~0.4 之间较常用）
    anchor_timeout_sec: float = 3.0     # 多少秒还没 anchor 就降级为 corr<=阈值触发


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


# -----------------------------
# 配置读写
# -----------------------------
def load_or_create_config() -> ClickConfig:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = ClickConfig()
        base_dict = asdict(base)
        base_dict.update(data)
        return ClickConfig(**base_dict)
    except Exception:
        cfg = ClickConfig()
        save_config(cfg)
        return cfg


def save_config(cfg: ClickConfig):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


# -----------------------------
# 截图/裁剪/解析/点击
# -----------------------------
def screenshot_full() -> np.ndarray:
    with mss() as sct:
        mon = sct.monitors[1]
        img = np.array(sct.grab(mon))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def get_screenshot_size() -> Tuple[int, int]:
    img = screenshot_full()
    h, w = img.shape[:2]
    return w, h


def crop_rect(img: np.ndarray, rect: Rect) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, min(rect.x, w - 1))
    y1 = max(0, min(rect.y, h - 1))
    x2 = max(0, min(rect.x + rect.w, w))
    y2 = max(0, min(rect.y + rect.h, h))
    if x2 <= x1 or y2 <= y1:
        return img[0:1, 0:1].copy()
    return img[y1:y2, x1:x2].copy()


def parse_remaining_seconds(text: str) -> Optional[int]:
    if not text:
        return None
    m = TIME_RE.search(text)
    if not m:
        return None
    minutes = m.group(1)
    seconds = m.group(2)
    mm = int(minutes) if minutes is not None else 0
    ss = int(seconds)
    return mm * 60 + ss


_mouse = Controller()


def click_rect(rect: Rect):
    x, y = rect.center()
    _mouse.position = (x, y)
    _mouse.click(Button.left, 1)


def precise_wait_until(target_perf_counter: float):
    while True:
        now = time.perf_counter()
        remain = target_perf_counter - now
        if remain <= 0:
            break
        if remain > 0.004:
            time.sleep(remain - 0.003)
        else:
            pass


# -----------------------------
# ROI 归一化存储/恢复
# -----------------------------
def rect_phys_to_norm(r: Rect, phys_w: int, phys_h: int) -> Dict[str, float]:
    phys_w = max(1, int(phys_w))
    phys_h = max(1, int(phys_h))
    return {
        "nx": r.x / phys_w,
        "ny": r.y / phys_h,
        "nw": r.w / phys_w,
        "nh": r.h / phys_h,
    }


def rect_norm_to_phys(d: Dict[str, float], phys_w: int, phys_h: int) -> Rect:
    phys_w = max(1, int(phys_w))
    phys_h = max(1, int(phys_h))
    x = int(round(d.get("nx", 0.0) * phys_w))
    y = int(round(d.get("ny", 0.0) * phys_h))
    w = int(round(d.get("nw", 0.0) * phys_w))
    h = int(round(d.get("nh", 0.0) * phys_h))
    w = max(1, w)
    h = max(1, h)
    return Rect(x, y, w, h)


# -----------------------------
# 全屏框选遮罩
# -----------------------------
class RoiOverlay(QWidget):
    finished = Signal(list)
    canceled = Signal()

    def __init__(self, instructions: List[str], parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setCursor(Qt.CrossCursor)

        self.instructions = instructions
        self.current_index = 0
        self.rects: List[Rect] = []

        self._dragging = False
        self._start = QPoint(0, 0)
        self._end = QPoint(0, 0)

        self.showFullScreen()
        self.raise_()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.canceled.emit()
            self.close()
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._start = event.position().toPoint()
            self._end = self._start
            self.update()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._end = event.position().toPoint()

            qrect = QRect(self._start, self._end).normalized()
            if qrect.width() < 5 or qrect.height() < 5:
                self.update()
                return

            r = Rect(qrect.x(), qrect.y(), qrect.width(), qrect.height())
            self.rects.append(r)
            self.current_index += 1

            if self.current_index >= len(self.instructions):
                self.finished.emit(self.rects)
                self.close()
            else:
                self._start = QPoint(0, 0)
                self._end = QPoint(0, 0)
                self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 90))

        pen_done = QPen(QColor(0, 255, 0, 220), 2)
        painter.setPen(pen_done)
        for idx, r in enumerate(self.rects):
            painter.drawRect(r.x, r.y, r.w, r.h)
            painter.setFont(QFont("Arial", 14))
            painter.drawText(r.x + 6, r.y + 20, f"#{idx+1}")

        if self._dragging:
            qrect = QRect(self._start, self._end).normalized()
            pen = QPen(QColor(255, 200, 0, 230), 2)
            painter.setPen(pen)
            painter.drawRect(qrect)

        painter.setPen(QPen(QColor(255, 255, 255, 240), 1))
        painter.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        tip = self.instructions[self.current_index] if self.current_index < len(self.instructions) else "完成"
        painter.drawText(24, 48, tip)

        painter.setFont(QFont("Microsoft YaHei", 12))
        painter.drawText(24, 78, "鼠标左键拖拽框选；ESC 取消。")


# -----------------------------
# 悬浮状态窗
# -----------------------------
class FloatingStatusWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_pos = None
        self._floating = True

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.text)

        self.resize(520, 320)
        self.apply_mode(True)

    def set_text(self, s: str):
        self.text.setPlainText(s)

    def apply_mode(self, floating: bool):
        self._floating = floating
        was_visible = self.isVisible()

        if floating:
            self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.setWindowOpacity(0.82)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WA_ShowWithoutActivating, True)

            self.setStyleSheet("background: rgba(20,20,20,180); border-radius: 10px;")
            self.text.setStyleSheet("""
                QPlainTextEdit{
                    background: transparent;
                    border: none;
                    color: rgba(255,255,255,220);
                    font-family: Consolas, 'Microsoft YaHei UI';
                    font-size: 12px;
                }
            """)
        else:
            self.setWindowFlags(Qt.Window)
            self.setWindowOpacity(1.0)
            self.setAttribute(Qt.WA_TranslucentBackground, False)
            self.setAttribute(Qt.WA_ShowWithoutActivating, False)
            self.setStyleSheet("")
            self.text.setStyleSheet("")

        self.setWindowTitle("进程状态")
        self.hide()
        if was_visible:
            self.show()

    def mousePressEvent(self, event):
        if self._floating and event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._floating and self._drag_pos is not None and (event.buttons() & Qt.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)


# -----------------------------
# OCR Worker（关键：最新帧流水线 + Anchor触发 + fallback）
# -----------------------------
class OCRWorker(QObject):
    log = Signal(str)
    status = Signal(str, object, object, object, str, str)  # raw_text, parsed, stable, corrected, mode, progress_text
    stopped = Signal()

    def __init__(self):
        super().__init__()
        self.cfg = ClickConfig()

        self.time_rect: Optional[Rect] = None
        self.click1_rect: Optional[Rect] = None
        self.click2_rect: Optional[Rect] = None
        self.click3_rect: Optional[Rect] = None

        self._stop = False
        self.ocr = RapidOCR()  # 只在 worker(QThread)里用

        self.clicked1a = False
        self.clicked1b = False
        self.clicked2 = False
        self.clicked3 = False

        self.last_valid_sec: Optional[int] = None
        self._stable_count = 0

        self.sec_correction: float = 0.0
        self.calibrating: bool = False
        self.calib_t0: float = 0.0
        self.calib_r0: Optional[int] = None

        self.mode = "OCR"
        self.countdown_target: Optional[float] = None

        self._debug_saved_once = False

        # ----- 截图/预处理流水线 -----
        self._frame_q: "Queue[Tuple[float, np.ndarray]]" = Queue(maxsize=1)  # (ts, proc_img)
        self._stop_evt = threading.Event()
        self._cap_thread: Optional[threading.Thread] = None

        self._latest_lock = threading.Lock()
        self._latest_ts: float = 0.0
        self._latest_sec: Optional[int] = None
        self._latest_text: str = ""
        self._latest_cost_ms: float = 0.0
        self._latest_age_ms: float = 0.0

        # ===== Anchor & fallback（解决：连续使用滞后 / anchor偶发未设置）=====
        self.anchor_unlock_t: Optional[float] = None
        self.anchor_quality: float = 0.0
        self._start_perf: float = 0.0
        self._fallback_mode: bool = False

    def set_params(self, cfg: ClickConfig,
                   time_rect: Rect, click1_rect: Rect, click2_rect: Rect, click3_rect: Rect):
        self.cfg = cfg
        self.time_rect = time_rect
        self.click1_rect = click1_rect
        self.click2_rect = click2_rect
        self.click3_rect = click3_rect

        self._stop = False
        self._stop_evt.clear()

        self.clicked1a = self.clicked1b = self.clicked2 = self.clicked3 = False
        self.last_valid_sec = None
        self._stable_count = 0

        self.sec_correction = 0.0
        self.calibrating = False
        self.calib_t0 = 0.0
        self.calib_r0 = None

        self.mode = "OCR"
        self.countdown_target = None

        self._debug_saved_once = False

        with self._latest_lock:
            self._latest_ts = 0.0
            self._latest_sec = None
            self._latest_text = ""
            self._latest_cost_ms = 0.0
            self._latest_age_ms = 0.0

        # Anchor reset
        self.anchor_unlock_t = None
        self.anchor_quality = 0.0
        self._start_perf = time.perf_counter()
        self._fallback_mode = False

        # 清空队列，避免残留
        try:
            while True:
                self._frame_q.get_nowait()
        except Exception:
            pass

    def stop(self):
        self._stop = True
        self._stop_evt.set()

    def preprocess_for_ocr(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if self.cfg.binarize:
            _, bw = cv2.threshold(gray, self.cfg.threshold, 255, cv2.THRESH_BINARY)
            return bw
        return gray

    def _capture_loop(self):
        """
        截图/裁剪/预处理线程：Queue(maxsize=1) 只保留最新帧，防止积压导致延迟。
        """
        interval = max(0.02, float(self.cfg.ocr_interval_sec))
        next_t = time.perf_counter()

        while (not self._stop_evt.is_set()) and (not self._stop):
            try:
                full = screenshot_full()
                crop = crop_rect(full, self.time_rect)
                proc = self.preprocess_for_ocr(crop)

                if self.cfg.debug_save_crop and not self._debug_saved_once:
                    try:
                        cv2.imwrite("debug_crop.png", proc)
                        self._debug_saved_once = True
                        self.log.emit("[DEBUG] 已保存 debug_crop.png（用于检查 ROI 是否跑偏）")
                    except Exception as e:
                        self.log.emit(f"[DEBUG] 保存 debug_crop.png 失败: {e}")

                ts = time.perf_counter()

                # 丢弃旧帧，仅保留最新
                if self._frame_q.full():
                    try:
                        self._frame_q.get_nowait()
                    except Exception:
                        pass
                try:
                    self._frame_q.put_nowait((ts, proc))
                except Exception:
                    pass

            except Exception as e:
                self.log.emit(f"[ERROR] capture exception: {e}")
                time.sleep(0.05)

            next_t += interval
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    def update_stable_sec(self, sec: Optional[int]) -> Optional[int]:
        if sec is None:
            self._stable_count = 0
            return self.last_valid_sec

        if self.last_valid_sec is None:
            self._stable_count += 1
            if self._stable_count >= self.cfg.stable_required:
                self.last_valid_sec = sec
            return self.last_valid_sec

        # 大跳变拒绝（OCR误识别常见）
        if abs(sec - self.last_valid_sec) > 8:
            self._stable_count = 0
            return self.last_valid_sec

        self._stable_count += 1
        if self._stable_count >= self.cfg.stable_required:
            self.last_valid_sec = sec
        return self.last_valid_sec

    def corrected_remaining(self, stable_sec: Optional[int]) -> Optional[float]:
        if stable_sec is None:
            return None
        return float(stable_sec) + float(self.sec_correction)

    def start_calibration(self, stable_sec: Optional[int]):
        if not self.cfg.enable_calibration:
            return
        if stable_sec is None:
            return
        self.calibrating = True
        self.calib_t0 = time.perf_counter()
        self.calib_r0 = int(round(stable_sec + self.sec_correction))
        self.log.emit(f"[CAL] start: r0={self.calib_r0}s, window={self.cfg.calibration_window_sec:.2f}s")

    def maybe_finish_calibration(self, stable_sec: Optional[int]):
        if not self.calibrating or not self.cfg.enable_calibration:
            return
        if stable_sec is None or self.calib_r0 is None:
            return
        t1 = time.perf_counter()
        elapsed = t1 - self.calib_t0
        if elapsed < self.cfg.calibration_window_sec:
            return

        r0 = float(self.calib_r0)
        r1_raw = float(stable_sec)
        expected_r1 = r0 - elapsed
        new_corr = expected_r1 - r1_raw

        old = self.sec_correction
        self.sec_correction = new_corr

        self.calibrating = False
        self.calib_r0 = None
        self.log.emit(f"[CAL] done: elapsed={elapsed:.3f}s, new_correction={new_corr:+.3f}s (old {old:+.3f}s)")

    # ===== 新增：Anchor 更新（EMA + 突变重置）=====
    def update_anchor(self, ts_capture: float, corr_remaining: float):
        est_unlock = ts_capture + float(corr_remaining)

        if self.anchor_unlock_t is None:
            self.anchor_unlock_t = est_unlock
            self.anchor_quality = 0.2
            return

        prev = self.anchor_unlock_t
        err = abs(est_unlock - prev)

        # 突变：认为刷新/跳秒，直接重置（解决“连续使用明显滞后”）
        if err > 1.2:
            self.anchor_unlock_t = est_unlock
            self.anchor_quality = 0.2
            self.log.emit(f"[ANCHOR] reset (jump {err:.3f}s)")
            return

        alpha = float(getattr(self.cfg, "anchor_alpha", 0.25))
        alpha = max(0.01, min(0.95, alpha))
        self.anchor_unlock_t = (1 - alpha) * prev + alpha * est_unlock

        # 质量更新：越稳定越高
        if err < 0.25:
            self.anchor_quality = min(1.0, self.anchor_quality + 0.05)
        else:
            self.anchor_quality = max(0.0, self.anchor_quality - 0.05)

    def reset_after_refresh(self):
        """
        点击刷新后，倒计时可能跳变。
        这里强制清空 anchor / stable，避免后续一轮整体滞后。
        """
        self.anchor_unlock_t = None
        self.anchor_quality = 0.0
        self._start_perf = time.perf_counter()
        self._fallback_mode = False

        self.last_valid_sec = None
        self._stable_count = 0

        # 校准窗口重新开始（不会影响也可不重置，这里更稳）
        self.calibrating = False
        self.calib_r0 = None
        self.calib_t0 = 0.0

    def build_progress_text(self, raw_text, parsed, stable, corrected) -> str:
        interval = max(0.05, float(self.cfg.ocr_interval_sec))
        hz = 1.0 / interval

        with self._latest_lock:
            cost_ms = self._latest_cost_ms
            age_ms = self._latest_age_ms

        trigger_mode = "ANCHOR" if not self._fallback_mode else "FALLBACK(corr)"
        anchor_line = "-"
        if self.anchor_unlock_t is not None:
            remain_to_unlock = self.anchor_unlock_t - time.perf_counter()
            anchor_line = f"unlock_at(perf)={self.anchor_unlock_t:.3f}  remain≈{remain_to_unlock:.3f}s  q={self.anchor_quality:.2f}"

        ocr_line = (
            f"模式：{self.mode}   OCR目标：{hz:.1f}Hz (interval={interval:.3f}s)   TRIGGER：{trigger_mode}\n"
            f"raw：{raw_text if raw_text else '-'}\n"
            f"parsed：{parsed if parsed is not None else '-'}   stable：{stable if stable is not None else '-'}   "
            f"corrected：{('-' if corrected is None else f'{corrected:.2f}')}"
        )
        perf_line = f"OCR cost：{cost_ms:.1f} ms   frame age：{age_ms:.1f} ms（越小越不滞后）"
        anchor_show = f"ANCHOR：{anchor_line}"

        click_line = (
            f"点击进度：1a={'√' if self.clicked1a else '×'}  "
            f"1b={'√' if self.clicked1b else '×'}  "
            f"2={'√' if self.clicked2 else '×'}  "
            f"3={'√' if self.clicked3 else '×'}"
        )

        if self.cfg.enable_calibration:
            if self.calibrating:
                elapsed = time.perf_counter() - self.calib_t0
                remain = max(0.0, float(self.cfg.calibration_window_sec) - elapsed)
                r0 = self.calib_r0 if self.calib_r0 is not None else "-"
                cal_line = f"校准：进行中  r0={r0}  已过={elapsed:.2f}s  剩余={remain:.2f}s"
            else:
                cal_line = f"校准：待机  window={self.cfg.calibration_window_sec:.2f}s"
        else:
            cal_line = "校准：关闭"

        corr_line = f"correction：{self.sec_correction:+.3f}s"

        if self.mode == "COUNTDOWN":
            base = float(self.cfg.countdown_after_click2_sec)
            remain_ms = int(round((self.countdown_target - time.perf_counter()) * 1000)) if self.countdown_target else 0
            cd_line = f"COUNTDOWN：基准={base:.3f}s  offset={self.cfg.click3_offset_ms}ms  距离点击3≈{remain_ms}ms"
        else:
            cd_line = "COUNTDOWN：未开始（点击2后进入）"

        return "\n".join([ocr_line, perf_line, anchor_show, click_line, cal_line, corr_line, cd_line])

    def run(self):
        if None in (self.time_rect, self.click1_rect, self.click2_rect, self.click3_rect):
            self.log.emit("[ERROR] ROI 未完整设置（需要：时间、点1、点2、点3）。")
            self.stopped.emit()
            return

        interval = max(0.05, float(self.cfg.ocr_interval_sec))
        self.log.emit(f"[INFO] start OCR loop, interval={interval:.3f}s (~{1/interval:.1f}Hz)")

        # 启动截图线程（流水线）
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cap_thread.start()
        self.log.emit("[INFO] capture thread started (latest-frame queue enabled).")

        try:
            while not self._stop:
                if self.mode == "OCR":
                    # 从队列取最新帧（没有就继续）
                    try:
                        ts, proc = self._frame_q.get(timeout=0.25)
                    except Empty:
                        progress = self.build_progress_text("", None, None, None)
                        self.status.emit("", None, None, None, self.mode, progress)
                        continue

                    # OCR 推理（在 worker 线程进行）
                    o0 = time.perf_counter()
                    result, _ = self.ocr(proc)
                    cost_ms = (time.perf_counter() - o0) * 1000.0

                    raw_text = ""
                    if result:
                        best = max(result, key=lambda r: len(r[1]) if r and len(r) >= 2 else 0)
                        raw_text = best[1] if best and len(best) >= 2 else ""

                    sec = parse_remaining_seconds(raw_text)

                    # 帧延迟：当前时刻 - 截图时刻
                    age_ms = (time.perf_counter() - ts) * 1000.0
                    with self._latest_lock:
                        self._latest_ts = ts
                        self._latest_sec = sec
                        self._latest_text = raw_text
                        self._latest_cost_ms = cost_ms
                        self._latest_age_ms = age_ms

                    stable = self.update_stable_sec(sec)
                    self.maybe_finish_calibration(stable)
                    corr = self.corrected_remaining(stable)

                    # === Anchor 更新：允许用 stable 或 sec 兜底（解决“偶发anchor未设置”）===
                    sec_for_anchor = stable if stable is not None else sec
                    corr_for_anchor = None
                    if sec_for_anchor is not None:
                        corr_for_anchor = float(sec_for_anchor) + float(self.sec_correction)

                    if corr_for_anchor is not None:
                        self.update_anchor(ts, corr_for_anchor)

                    # === Anchor 超时降级（防止完全不触发）===
                    if (self.anchor_unlock_t is None) and (not self._fallback_mode):
                        if (time.perf_counter() - self._start_perf) > float(self.cfg.anchor_timeout_sec):
                            self._fallback_mode = True
                            self.log.emit("[ANCHOR] timeout -> fallback to corr-threshold triggers")

                    progress = self.build_progress_text(raw_text, sec, stable, corr)
                    self.status.emit(raw_text, sec, stable, corr, self.mode, progress)

                    now = time.perf_counter()

                    # ========== 触发策略 ==========
                    if (not self._fallback_mode) and (self.anchor_unlock_t is not None) and (self.anchor_quality >= 0.15):
                        # 动态计划：每帧都按当前 anchor 计算触发时刻，避免漂移
                        t1a_at = self.anchor_unlock_t - float(self.cfg.trigger1a_sec)
                        t1b_at = self.anchor_unlock_t - float(self.cfg.trigger1b_sec)
                        t2_at = self.anchor_unlock_t - float(self.cfg.trigger2_sec)

                        if (not self.clicked1a) and now >= t1a_at:
                            self.log.emit("[TRIGGER] anchor -> click1 (1st)")
                            click_rect(self.click1_rect)
                            self.clicked1a = True
                            self.start_calibration(stable)
                            self.reset_after_refresh()

                        if self.clicked1a and (not self.clicked1b) and now >= t1b_at:
                            self.log.emit("[TRIGGER] anchor -> click1 (2nd)")
                            click_rect(self.click1_rect)
                            self.clicked1b = True
                            self.start_calibration(stable)
                            self.reset_after_refresh()

                        if (not self.clicked2) and now >= t2_at:
                            self.log.emit("[TRIGGER] anchor -> click2 (enter submenu)")
                            click_rect(self.click2_rect)
                            self.clicked2 = True

                            # 点2后：停止截图/OCR，进入内部倒计时
                            self._stop_evt.set()

                            base = float(self.cfg.countdown_after_click2_sec)
                            offset = float(self.cfg.click3_offset_ms) / 1000.0
                            self.countdown_target = time.perf_counter() + base + offset
                            self.mode = "COUNTDOWN"
                            self.log.emit(f"[COUNTDOWN] now stop OCR. target in {base:+.3f}s + offset {offset:+.3f}s")

                    else:
                        # 降级模式：沿用 corr<=阈值触发（只要 corr 有，就能执行）
                        if corr is not None:
                            if (not self.clicked1a) and corr <= self.cfg.trigger1a_sec:
                                self.log.emit(f"[TRIGGER] fallback corr={corr:.2f}s <= t1a -> click1 (1st)")
                                click_rect(self.click1_rect)
                                self.clicked1a = True
                                self.start_calibration(stable)
                                self.reset_after_refresh()

                            if self.clicked1a and (not self.clicked1b) and corr <= self.cfg.trigger1b_sec:
                                self.log.emit(f"[TRIGGER] fallback corr={corr:.2f}s <= t1b -> click1 (2nd)")
                                click_rect(self.click1_rect)
                                self.clicked1b = True
                                self.start_calibration(stable)
                                self.reset_after_refresh()

                            if (not self.clicked2) and corr <= self.cfg.trigger2_sec:
                                self.log.emit(f"[TRIGGER] fallback corr={corr:.2f}s <= t2 -> click2 (enter submenu)")
                                click_rect(self.click2_rect)
                                self.clicked2 = True

                                self._stop_evt.set()
                                base = float(self.cfg.countdown_after_click2_sec)
                                offset = float(self.cfg.click3_offset_ms) / 1000.0
                                self.countdown_target = time.perf_counter() + base + offset
                                self.mode = "COUNTDOWN"
                                self.log.emit(f"[COUNTDOWN] now stop OCR. target in {base:+.3f}s + offset {offset:+.3f}s")

                else:
                    progress = self.build_progress_text("", None, None, None)
                    self.status.emit("", None, None, None, self.mode, progress)

                    if self.countdown_target is None:
                        self.log.emit("[ERROR] countdown_target missing.")
                        self._stop = True
                        break

                    if not self.clicked3:
                        precise_wait_until(self.countdown_target)
                        self.log.emit("[TRIGGER] click3 now.")
                        click_rect(self.click3_rect)
                        self.clicked3 = True
                        self._stop = True
                        break

        finally:
            # 确保线程退出
            self._stop_evt.set()
            if self._cap_thread and self._cap_thread.is_alive():
                self._cap_thread.join(timeout=0.8)

            self.log.emit("[INFO] worker stopped.")
            self.stopped.emit()


# -----------------------------
# 主窗口 UI（含：屏幕参数/ROI保存加载/悬浮窗开关）
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR 模拟点击器（ROI保存 + 分辨率/缩放适配 + 最新帧OCR流水线 + Anchor稳定触发）")
        self.resize(1120, 860)

        self.cfg = load_or_create_config()

        auto_phys_w, auto_phys_h = get_screenshot_size()
        self.phys_w = auto_phys_w
        self.phys_h = auto_phys_h
        self.scale_x, self.scale_y = self._compute_scale_xy(auto_phys_w, auto_phys_h)

        self.time_rect: Optional[Rect] = None
        self.click1_rect: Optional[Rect] = None
        self.click2_rect: Optional[Rect] = None
        self.click3_rect: Optional[Rect] = None

        self.worker = OCRWorker()
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.status.connect(self.on_status)
        self.worker.stopped.connect(self.on_stopped)

        self.float_win = FloatingStatusWindow()
        self.float_win.move(30, 30)
        self.float_win.show()

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left = QVBoxLayout()
        right = QVBoxLayout()
        layout.addLayout(left, 1)
        layout.addLayout(right, 1)

        # 屏幕参数
        screen_box = QGroupBox("屏幕参数（自动/手动适配）")
        screen_form = QFormLayout(screen_box)

        self.chk_manual_screen = QCheckBox("使用手动屏幕参数（自动不准时再开）")
        self.chk_manual_screen.setChecked(self.cfg.use_manual_screen)

        self.spin_phys_w = QSpinBox()
        self.spin_phys_w.setRange(640, 20000)
        self.spin_phys_w.setValue(self.cfg.manual_phys_w)

        self.spin_phys_h = QSpinBox()
        self.spin_phys_h.setRange(480, 20000)
        self.spin_phys_h.setValue(self.cfg.manual_phys_h)

        self.spin_scale_pct = QSpinBox()
        self.spin_scale_pct.setRange(50, 400)
        self.spin_scale_pct.setValue(self.cfg.manual_scale_percent)

        self.btn_apply_screen = QPushButton("应用屏幕参数")
        self.btn_apply_screen.clicked.connect(self.apply_screen_params)

        self.lbl_screen_info = QLabel("")
        self.lbl_screen_info.setWordWrap(True)

        screen_form.addRow(self.chk_manual_screen, QLabel(""))
        screen_form.addRow("物理分辨率 W", self.spin_phys_w)
        screen_form.addRow("物理分辨率 H", self.spin_phys_h)
        screen_form.addRow("缩放倍率 %（200%→2.0）", self.spin_scale_pct)
        screen_form.addRow(self.btn_apply_screen, QLabel(""))
        screen_form.addRow("当前生效参数", self.lbl_screen_info)

        left.addWidget(screen_box)

        # ROI
        roi_box = QGroupBox("区域选择（两步）+ 保存/加载")
        roi_layout = QVBoxLayout(roi_box)

        self.btn_select_roi_3 = QPushButton("选择区域（时间 + 点击1 + 点击2）")
        self.btn_select_roi_3.clicked.connect(self.select_roi_step1)

        self.btn_select_roi_1 = QPushButton("选择区域（点击3：请先打开下级菜单再点这里）")
        self.btn_select_roi_1.clicked.connect(self.select_roi_step2)

        self.btn_load_roi = QPushButton("加载已保存 ROI")
        self.btn_load_roi.clicked.connect(lambda: self.load_saved_roi_to_rects(silent=False))

        self.btn_clear_roi = QPushButton("清除已保存 ROI")
        self.btn_clear_roi.clicked.connect(self.clear_saved_roi)

        self.lbl_roi = QLabel("未选择 ROI")
        self.lbl_roi.setWordWrap(True)

        roi_layout.addWidget(self.btn_select_roi_3)
        roi_layout.addWidget(self.btn_select_roi_1)
        row_roi = QHBoxLayout()
        row_roi.addWidget(self.btn_load_roi)
        row_roi.addWidget(self.btn_clear_roi)
        roi_layout.addLayout(row_roi)
        roi_layout.addWidget(self.lbl_roi)

        left.addWidget(roi_box)

        # 参数配置
        cfg_box = QGroupBox(f"参数配置（保存到 {CONFIG_PATH}）")
        form = QFormLayout(cfg_box)

        self.spin_interval = QDoubleSpinBox()
        self.spin_interval.setRange(0.05, 1.0)
        self.spin_interval.setDecimals(3)
        self.spin_interval.setSingleStep(0.005)
        self.spin_interval.setValue(self.cfg.ocr_interval_sec)

        self.spin_t1a = QSpinBox()
        self.spin_t1a.setRange(0, 9999)
        self.spin_t1a.setValue(self.cfg.trigger1a_sec)

        self.spin_t1b = QSpinBox()
        self.spin_t1b.setRange(0, 9999)
        self.spin_t1b.setValue(self.cfg.trigger1b_sec)

        self.spin_t2 = QSpinBox()
        self.spin_t2.setRange(0, 9999)
        self.spin_t2.setValue(self.cfg.trigger2_sec)

        self.spin_cd = QDoubleSpinBox()
        self.spin_cd.setRange(0.0, 60.0)
        self.spin_cd.setDecimals(3)
        self.spin_cd.setSingleStep(0.05)
        self.spin_cd.setValue(self.cfg.countdown_after_click2_sec)

        self.spin_offset = QSpinBox()
        self.spin_offset.setRange(-600000, 600000)
        self.spin_offset.setValue(self.cfg.click3_offset_ms)

        self.chk_binarize = QCheckBox("二值化")
        self.chk_binarize.setChecked(self.cfg.binarize)

        self.spin_thresh = QSpinBox()
        self.spin_thresh.setRange(0, 255)
        self.spin_thresh.setValue(self.cfg.threshold)

        self.spin_stable = QSpinBox()
        self.spin_stable.setRange(1, 10)
        self.spin_stable.setValue(self.cfg.stable_required)

        self.chk_cal = QCheckBox("启用刷新后校准")
        self.chk_cal.setChecked(self.cfg.enable_calibration)

        self.spin_calwin = QDoubleSpinBox()
        self.spin_calwin.setRange(0.5, 5.0)
        self.spin_calwin.setDecimals(2)
        self.spin_calwin.setSingleStep(0.1)
        self.spin_calwin.setValue(self.cfg.calibration_window_sec)

        self.chk_debug_crop = QCheckBox("调试：保存一次 OCR 裁剪图(debug_crop.png)")
        self.chk_debug_crop.setChecked(self.cfg.debug_save_crop)

        form.addRow("OCR 间隔秒（0.125~0.167≈6~8Hz）", self.spin_interval)
        form.addRow("点击1 第一次触发（<= 秒）", self.spin_t1a)
        form.addRow("点击1 第二次触发（<= 秒）", self.spin_t1b)
        form.addRow("点击2 触发（<= 秒）", self.spin_t2)
        form.addRow("点击2后内部倒计时基准（秒）", self.spin_cd)
        form.addRow("点击3 微调 offset（ms，可正可负）", self.spin_offset)
        form.addRow(self.chk_binarize, QLabel(""))
        form.addRow("二值化阈值（0-255）", self.spin_thresh)
        form.addRow("稳定要求（连续N次）", self.spin_stable)
        form.addRow(self.chk_cal, QLabel(""))
        form.addRow("校准窗口（秒）", self.spin_calwin)
        form.addRow(self.chk_debug_crop, QLabel(""))

        left.addWidget(cfg_box)

        # 控制区
        ctrl_box = QGroupBox("控制")
        ctrl_layout = QHBoxLayout(ctrl_box)

        self.btn_save = QPushButton("保存配置")
        self.btn_save.clicked.connect(self.save_cfg)

        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)

        self.chk_floating = QCheckBox("悬浮窗(半透明置顶)")
        self.chk_floating.setChecked(True)
        self.chk_floating.stateChanged.connect(self.on_toggle_floating)

        ctrl_layout.addWidget(self.btn_save)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.chk_floating)

        left.addWidget(ctrl_box)
        left.addStretch(1)

        # 状态区
        status_box = QGroupBox("实时状态")
        status_layout = QVBoxLayout(status_box)
        self.lbl_mode = QLabel("模式：-")
        self.lbl_raw = QLabel("OCR文本：-")
        self.lbl_parsed = QLabel("解析秒：-")
        self.lbl_stable = QLabel("稳定秒：-")
        self.lbl_corrected = QLabel("校准后秒：-")
        for lab in (self.lbl_mode, self.lbl_raw, self.lbl_parsed, self.lbl_stable, self.lbl_corrected):
            lab.setWordWrap(True)
            status_layout.addWidget(lab)
        right.addWidget(status_box)

        # 日志区
        log_box = QGroupBox("日志")
        log_layout = QVBoxLayout(log_box)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        log_layout.addWidget(self.log)
        right.addWidget(log_box, 1)

        self.append_log(f"[INFO] 配置文件：{CONFIG_PATH}")
        self.update_screen_info_label()

        # 启动时自动加载 ROI
        self.load_saved_roi_to_rects(silent=True)
        self.refresh_roi_label()

    # --------- 屏幕适配 ---------
    def _compute_scale_xy(self, phys_w: int, phys_h: int) -> Tuple[float, float]:
        screen = QApplication.primaryScreen()
        logical_w = max(1, int(screen.geometry().width()))
        logical_h = max(1, int(screen.geometry().height()))
        sx = phys_w / float(logical_w)
        sy = phys_h / float(logical_h)
        return sx, sy

    def apply_screen_params(self):
        self.cfg.use_manual_screen = bool(self.chk_manual_screen.isChecked())
        self.cfg.manual_phys_w = int(self.spin_phys_w.value())
        self.cfg.manual_phys_h = int(self.spin_phys_h.value())
        self.cfg.manual_scale_percent = int(self.spin_scale_pct.value())
        save_config(self.cfg)

        if self.cfg.use_manual_screen:
            self.phys_w = max(1, self.cfg.manual_phys_w)
            self.phys_h = max(1, self.cfg.manual_phys_h)
            s = max(0.5, min(4.0, self.cfg.manual_scale_percent / 100.0))
            self.scale_x = s
            self.scale_y = s
            self.append_log(f"[SCREEN] 手动：phys={self.phys_w}x{self.phys_h}, scale={s:.3f}")
        else:
            auto_w, auto_h = get_screenshot_size()
            self.phys_w, self.phys_h = auto_w, auto_h
            self.scale_x, self.scale_y = self._compute_scale_xy(auto_w, auto_h)
            self.append_log(f"[SCREEN] 自动：phys={self.phys_w}x{self.phys_h}, scale≈({self.scale_x:.3f},{self.scale_y:.3f})")

        self.update_screen_info_label()
        self.load_saved_roi_to_rects(silent=False)
        self.refresh_roi_label()

    def update_screen_info_label(self):
        mode = "手动" if self.cfg.use_manual_screen else "自动"
        self.lbl_screen_info.setText(
            f"{mode} | 物理分辨率={self.phys_w}×{self.phys_h} | scale≈({self.scale_x:.3f},{self.scale_y:.3f})\n"
            f"提示：4K 200% 通常 scale≈2.0；自动不准再切手动。"
        )

    def _scale_rect_logical_to_phys(self, r: Rect) -> Rect:
        return Rect(
            int(round(r.x * self.scale_x)),
            int(round(r.y * self.scale_y)),
            int(round(r.w * self.scale_x)),
            int(round(r.h * self.scale_y)),
        )

    # --------- ROI 保存/加载 ---------
    def save_current_roi_to_config(self):
        if None in (self.time_rect, self.click1_rect, self.click2_rect, self.click3_rect):
            return
        self.cfg.roi_time = rect_phys_to_norm(self.time_rect, self.phys_w, self.phys_h)
        self.cfg.roi_click1 = rect_phys_to_norm(self.click1_rect, self.phys_w, self.phys_h)
        self.cfg.roi_click2 = rect_phys_to_norm(self.click2_rect, self.phys_w, self.phys_h)
        self.cfg.roi_click3 = rect_phys_to_norm(self.click3_rect, self.phys_w, self.phys_h)
        save_config(self.cfg)
        self.append_log("[ROI] 已保存四个区域（归一化坐标）。")

    def load_saved_roi_to_rects(self, silent: bool = False):
        if not (self.cfg.roi_time and self.cfg.roi_click1 and self.cfg.roi_click2 and self.cfg.roi_click3):
            if not silent:
                self.append_log("[ROI] 配置中没有完整 ROI。")
            return
        try:
            self.time_rect = rect_norm_to_phys(self.cfg.roi_time, self.phys_w, self.phys_h)
            self.click1_rect = rect_norm_to_phys(self.cfg.roi_click1, self.phys_w, self.phys_h)
            self.click2_rect = rect_norm_to_phys(self.cfg.roi_click2, self.phys_w, self.phys_h)
            self.click3_rect = rect_norm_to_phys(self.cfg.roi_click3, self.phys_w, self.phys_h)
            if not silent:
                self.append_log("[ROI] 已从配置恢复四个区域。")
        except Exception as e:
            self.append_log(f"[ROI] 恢复失败：{e}")

    def clear_saved_roi(self):
        self.cfg.roi_time = None
        self.cfg.roi_click1 = None
        self.cfg.roi_click2 = None
        self.cfg.roi_click3 = None
        save_config(self.cfg)

        self.time_rect = None
        self.click1_rect = None
        self.click2_rect = None
        self.click3_rect = None

        self.append_log("[ROI] 已清除保存 ROI，并清空当前 ROI。")
        self.refresh_roi_label()

    # --------- UI 通用 ---------
    def on_toggle_floating(self):
        self.float_win.apply_mode(self.chk_floating.isChecked())
        self.float_win.show()
        self.float_win.raise_()

    def append_log(self, s: str):
        self.log.appendPlainText(s)

    def cfg_from_ui(self) -> ClickConfig:
        cfg = self.cfg
        cfg.ocr_interval_sec = float(self.spin_interval.value())
        cfg.trigger1a_sec = int(self.spin_t1a.value())
        cfg.trigger1b_sec = int(self.spin_t1b.value())
        cfg.trigger2_sec = int(self.spin_t2.value())
        cfg.countdown_after_click2_sec = float(self.spin_cd.value())
        cfg.click3_offset_ms = int(self.spin_offset.value())
        cfg.binarize = bool(self.chk_binarize.isChecked())
        cfg.threshold = int(self.spin_thresh.value())
        cfg.stable_required = int(self.spin_stable.value())
        cfg.enable_calibration = bool(self.chk_cal.isChecked())
        cfg.calibration_window_sec = float(self.spin_calwin.value())
        cfg.debug_save_crop = bool(self.chk_debug_crop.isChecked())

        cfg.use_manual_screen = bool(self.chk_manual_screen.isChecked())
        cfg.manual_phys_w = int(self.spin_phys_w.value())
        cfg.manual_phys_h = int(self.spin_phys_h.value())
        cfg.manual_scale_percent = int(self.spin_scale_pct.value())
        return cfg

    def save_cfg(self):
        self.cfg = self.cfg_from_ui()
        save_config(self.cfg)
        self.append_log("[INFO] 配置已保存。")

    def refresh_roi_label(self):
        def fmt(r: Optional[Rect]):
            if r is None:
                return "未设置"
            return f"({r.x},{r.y},{r.w},{r.h})"
        saved_ok = all([self.cfg.roi_time, self.cfg.roi_click1, self.cfg.roi_click2, self.cfg.roi_click3])
        saved_tip = "已保存" if saved_ok else "未保存"
        self.lbl_roi.setText(
            f"[当前物理ROI | {saved_tip}]\n"
            f"time: {fmt(self.time_rect)}\n"
            f"click1: {fmt(self.click1_rect)}\n"
            f"click2: {fmt(self.click2_rect)}\n"
            f"click3: {fmt(self.click3_rect)}"
        )

    # --------- ROI 框选 ---------
    def select_roi_step1(self):
        if self.thread.isRunning():
            QMessageBox.information(self, "提示", "请先 Stop 再重新选择 ROI。")
            return
        instructions = [
            "Step1/3：框选【时间识别区域】",
            "Step2/3：框选【点击区域1（刷新按钮）】",
            "Step3/3：框选【点击区域2】",
        ]
        self.overlay = RoiOverlay(instructions)
        self.overlay.finished.connect(self.on_roi_step1_finished)
        self.overlay.canceled.connect(lambda: self.append_log("[INFO] ROI Step1 已取消。"))

    def on_roi_step1_finished(self, rects: List[Rect]):
        if len(rects) != 3:
            self.append_log("[ERROR] Step1 ROI 数量不正确。")
            return
        self.time_rect = self._scale_rect_logical_to_phys(rects[0])
        self.click1_rect = self._scale_rect_logical_to_phys(rects[1])
        self.click2_rect = self._scale_rect_logical_to_phys(rects[2])
        self.append_log("[INFO] Step1 ROI 已设置（逻辑->物理已换算）。")

        if self.click3_rect is not None:
            self.save_current_roi_to_config()
        self.refresh_roi_label()

    def select_roi_step2(self):
        if self.thread.isRunning():
            QMessageBox.information(self, "提示", "请先 Stop 再重新选择 ROI。")
            return
        instructions = ["Step1/1：框选【点击区域3】（请确保已打开下级菜单）"]
        self.overlay = RoiOverlay(instructions)
        self.overlay.finished.connect(self.on_roi_step2_finished)
        self.overlay.canceled.connect(lambda: self.append_log("[INFO] ROI Step2 已取消。"))

    def on_roi_step2_finished(self, rects: List[Rect]):
        if len(rects) != 1:
            self.append_log("[ERROR] Step2 ROI 数量不正确。")
            return
        self.click3_rect = self._scale_rect_logical_to_phys(rects[0])
        self.append_log("[INFO] Step2 ROI 已设置（逻辑->物理已换算）。")

        if None not in (self.time_rect, self.click1_rect, self.click2_rect, self.click3_rect):
            self.save_current_roi_to_config()
        self.refresh_roi_label()

    # --------- 运行控制 ---------
    def start(self):
        # 如果当前没 ROI，尝试从配置恢复
        if None in (self.time_rect, self.click1_rect, self.click2_rect, self.click3_rect):
            self.load_saved_roi_to_rects(silent=True)

        if None in (self.time_rect, self.click1_rect, self.click2_rect, self.click3_rect):
            QMessageBox.warning(self, "缺少ROI", "需要先设置四个 ROI（或加载已保存 ROI）。")
            return

        if self.thread.isRunning():
            QMessageBox.information(self, "提示", "已经在运行中。")
            return

        self.save_cfg()

        if not (self.cfg.trigger1a_sec >= self.cfg.trigger1b_sec >= self.cfg.trigger2_sec):
            self.append_log("[WARN] 建议满足：t1a >= t1b >= t2（否则触发顺序可能不符合预期）。")

        self.worker.set_params(self.cfg, self.time_rect, self.click1_rect, self.click2_rect, self.click3_rect)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_select_roi_3.setEnabled(False)
        self.btn_select_roi_1.setEnabled(False)

        self.append_log("[INFO] 启动。")
        self.thread.start()

    def stop(self):
        if self.thread.isRunning():
            self.append_log("[INFO] 请求停止...")
            self.worker.stop()
        else:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_select_roi_3.setEnabled(True)
            self.btn_select_roi_1.setEnabled(True)

    def on_stopped(self):
        self.thread.quit()
        self.thread.wait(1500)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_select_roi_3.setEnabled(True)
        self.btn_select_roi_1.setEnabled(True)
        self.append_log("[INFO] 已停止。")

    def on_status(self, raw_text: str, parsed, stable, corrected, mode: str, progress_text: str):
        self.lbl_mode.setText(f"模式：{mode}")
        self.lbl_raw.setText(f"OCR文本：{raw_text if raw_text else '-'}")
        self.lbl_parsed.setText(f"解析秒：{parsed if parsed is not None else '-'}")
        self.lbl_stable.setText(f"稳定秒：{stable if stable is not None else '-'}")
        self.lbl_corrected.setText(f"校准后秒：{(f'{corrected:.2f}' if corrected is not None else '-')}")
        self.float_win.set_text(progress_text if progress_text else "")

    def closeEvent(self, event):
        try:
            if self.thread.isRunning():
                self.worker.stop()
                self.thread.quit()
                self.thread.wait(1500)
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
