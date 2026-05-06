# ui/dashboard.py
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import threading
import queue
import time
import os

from core.motion_detector import MotionDetector
from core.object_tracker  import CentroidTracker
from core.classifier      import ObjectClassifier
from core.video_handler   import VideoSource, IncidentRecorder
from utils.annotator      import Annotator
from utils.config         import (
    DASHBOARD_THEME as T, ALERT_COLOURS,
    MOG2_VAR_THRESHOLD, MIN_CONTOUR_AREA,
    VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT
)

BG    = "#0A0A0F"
PANEL = "#13131A"
BORD  = "#1E1E2E"
TEXT  = "#E0E0F0"
SUB   = "#555570"
ACC   = "#00FFA3"
DNG   = "#FF4D6D"
WRN   = "#FFB547"
BLK   = "#07070D"


class Dashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart Security Camera — CSY3058")
        self.root.configure(bg=BG)
        self.root.geometry("1380x820")

        self.detector   = MotionDetector()
        self.tracker    = CentroidTracker()
        self.classifier = ObjectClassifier()
        self.annotator  = Annotator()
        self.recorder   = None

        self._running        = False
        self._thread         = None
        self._source         = VIDEO_SOURCE
        self._frame_queue    = queue.Queue(maxsize=2)
        self._alert_queue    = queue.Queue()
        self._sensitivity    = tk.DoubleVar(value=MOG2_VAR_THRESHOLD)
        self._min_area       = tk.IntVar(value=MIN_CONTOUR_AREA)
        self._show_mask      = tk.BooleanVar(value=False)
        self._hog_enabled    = tk.BooleanVar(value=True)
        self._fps_var        = tk.StringVar(value="-- FPS")
        self._status_var     = tk.StringVar(value="● IDLE")
        self._source_var     = tk.StringVar(value="Webcam (index 0)")
        self._alerts         = 0
        self._humans         = 0
        self._vehicles       = 0
        self._clips          = 0
        self._imgtk          = None  # MUST keep reference

        self._build_ui()
        self.root.after(200, self._refresh_clips)
        self.root.after(33,  self._poll_frame)   # 30fps UI poll
        self.root.after(100, self._poll_alerts)

    # ═════════════════════════════════════════════════════════════════
    # UI BUILD
    # ═════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=BLK, height=50)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="◈ SMART SECURITY CAMERA",
                 bg=BLK, fg=ACC,
                 font=("Courier New", 12, "bold")).pack(side="left", padx=16, pady=12)
        tk.Label(hdr, text="CSY3058 · Media Technology",
                 bg=BLK, fg=SUB, font=("Courier New", 8)).pack(side="left")
        tk.Label(hdr, textvariable=self._fps_var,
                 bg=BLK, fg=ACC, font=("Courier New", 11, "bold")).pack(side="right", padx=16)
        tk.Label(hdr, textvariable=self._status_var,
                 bg=BLK, fg=SUB, font=("Courier New", 9)).pack(side="right", padx=4)

        # Toolbar
        tb = tk.Frame(self.root, bg=BORD, height=44)
        tb.pack(fill="x"); tb.pack_propagate(False)
        self._btn(tb, "▶ START",        self._start,         ACC,   "#000").pack(side="left", padx=(10,3), pady=6)
        self._btn(tb, "■ STOP",         self._stop,          DNG,   "#fff").pack(side="left", padx=3,     pady=6)
        self._btn(tb, "📁 Load Video",  self._load_video,    PANEL, TEXT).pack(side="left",  padx=3,     pady=6)
        self._btn(tb, "🔴 Webcam",      self._use_webcam,    PANEL, TEXT).pack(side="left",  padx=3,     pady=6)
        self._btn(tb, "↺ Refresh",      self._refresh_clips, PANEL, TEXT).pack(side="left",  padx=3,     pady=6)
        self._btn(tb, "⟳ Reset BG",    self._reset_bg,      PANEL, WRN).pack(side="left",   padx=3,     pady=6)
        tk.Label(tb, textvariable=self._source_var,
                 bg=BORD, fg=SUB, font=("Courier New", 7)).pack(side="right", padx=12)

        # Body — grid layout for reliable sizing
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=6)
        body.columnconfigure(0, weight=1)   # left expands
        body.columnconfigure(1, minsize=330, weight=0)  # right fixed
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, minsize=130, weight=0)

        # ── Feed (row 0, col 0) ──────────────────────────────────────
        feed_outer = tk.Frame(body, bg=BORD)
        feed_outer.grid(row=0, column=0, sticky="nsew", padx=(0,5), pady=(0,5))
        tk.Label(feed_outer, text="  ● LIVE FEED — CAM 01",
                 bg=BORD, fg=SUB,
                 font=("Courier New", 7, "bold"), pady=4).pack(anchor="w", fill="x")
        feed_inner = tk.Frame(feed_outer, bg=PANEL)
        feed_inner.pack(fill="both", expand=True, padx=1, pady=(0,1))

        self.feed_label = tk.Label(feed_inner, bg="#000",
                                   text="Press  ▶ START  to begin",
                                   fg=SUB, font=("Courier New", 12))
        self.feed_label.pack(fill="both", expand=True)

        # ── Clips (row 1, col 0) ──────────────────────────────────────
        clips_outer = tk.Frame(body, bg=BORD)
        clips_outer.grid(row=1, column=0, sticky="nsew", padx=(0,5))
        tk.Label(clips_outer, text="  📼  SAVED INCIDENT CLIPS",
                 bg=BORD, fg=SUB,
                 font=("Courier New", 7, "bold"), pady=4).pack(anchor="w", fill="x")
        clips_inner = tk.Frame(clips_outer, bg=PANEL)
        clips_inner.pack(fill="both", expand=True, padx=1, pady=(0,1))

        row = tk.Frame(clips_inner, bg=PANEL)
        row.pack(fill="both", expand=True, padx=5, pady=5)
        self.clips_list = tk.Listbox(row, bg=BG, fg=TEXT,
                                     font=("Courier New", 8), relief="flat",
                                     selectbackground=ACC, selectforeground="#000",
                                     activestyle="none", bd=0, height=4)
        self.clips_list.pack(side="left", fill="both", expand=True)
        csb = tk.Scrollbar(row, orient="vertical", command=self.clips_list.yview,
                           bg=PANEL, troughcolor=BG)
        csb.pack(side="left", fill="y")
        self.clips_list.config(yscrollcommand=csb.set)
        self._btn(row, "▶ Play", self._play_clip, ACC, "#000").pack(side="left", padx=8)

        # ── Right column (rows 0+1, col 1) ───────────────────────────
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")

        # Stats
        stats = tk.Frame(right, bg=BG)
        stats.pack(fill="x", pady=(0,5))
        self._stat = {}
        for key, lbl, clr in [
            ("alerts",   "ALERTS",   DNG),
            ("humans",   "HUMANS",   ACC),
            ("vehicles", "VEHICLES", WRN),
            ("clips",    "CLIPS",    SUB),
        ]:
            cell = tk.Frame(stats, bg=PANEL)
            cell.pack(side="left", expand=True, fill="both", padx=2)
            n = tk.Label(cell, text="0", bg=PANEL, fg=clr,
                         font=("Courier New", 20, "bold"))
            n.pack(pady=(10,0))
            tk.Label(cell, text=lbl, bg=PANEL, fg=SUB,
                     font=("Courier New", 6, "bold")).pack(pady=(0,10))
            self._stat[key] = n

        # Alert log
        log_outer = tk.Frame(right, bg=BORD)
        log_outer.pack(fill="both", expand=True, pady=(0,5))
        tk.Label(log_outer, text="  🔔  ALERT LOG",
                 bg=BORD, fg=SUB,
                 font=("Courier New", 7, "bold"), pady=4).pack(anchor="w", fill="x")
        log_inner = tk.Frame(log_outer, bg=PANEL)
        log_inner.pack(fill="both", expand=True, padx=1, pady=(0,1))
        self.alert_list = tk.Listbox(log_inner, bg=BG, fg=TEXT,
                                     font=("Courier New", 8), relief="flat",
                                     selectbackground=ACC,
                                     activestyle="none", bd=0)
        self.alert_list.pack(fill="both", expand=True, padx=4, pady=(4,0))
        self._btn(log_inner, "Clear Log",
                  lambda: self.alert_list.delete(0, "end"),
                  BORD, SUB).pack(pady=4)

        # Settings
        sett_outer = tk.Frame(right, bg=BORD)
        sett_outer.pack(fill="x")
        tk.Label(sett_outer, text="  ⚙  DETECTION SETTINGS",
                 bg=BORD, fg=SUB,
                 font=("Courier New", 7, "bold"), pady=4).pack(anchor="w", fill="x")
        sett_inner = tk.Frame(sett_outer, bg=PANEL)
        sett_inner.pack(fill="x", padx=1, pady=(0,1))
        self._build_settings(sett_inner)

        # Statusbar
        sb = tk.Frame(self.root, bg=BLK, height=20)
        sb.pack(fill="x", side="bottom"); sb.pack_propagate(False)
        tk.Label(sb,
                 text="CSY3058 Smart Security Camera  ·  University of Northampton  ·  Python + OpenCV + MOG2 + HOG",
                 bg=BLK, fg=SUB, font=("Courier New", 6)).pack(side="left", padx=10)

    def _build_settings(self, p):
        kw  = dict(padx=10, pady=2)
        lkw = dict(bg=PANEL, fg=SUB, font=("Courier New", 7))
        skw = dict(bg=PANEL, fg=TEXT, troughcolor=BG,
                   highlightthickness=0, bd=0, orient="horizontal",
                   font=("Courier New", 7))
        ckw = dict(bg=PANEL, fg=TEXT, selectcolor=BG,
                   activebackground=PANEL, font=("Courier New", 8), anchor="w")
        tk.Label(p, text="Sensitivity  (lower = more sensitive)", **lkw).pack(anchor="w", **kw)
        tk.Scale(p, variable=self._sensitivity, from_=10, to=150,
                 command=self._update_sensitivity, **skw).pack(fill="x", **kw)
        tk.Label(p, text="Min Object Size  (px²)", **lkw).pack(anchor="w", **kw)
        tk.Scale(p, variable=self._min_area, from_=500, to=15000, **skw).pack(fill="x", **kw)
        tk.Checkbutton(p, text=" Show foreground mask",
                       variable=self._show_mask, **ckw).pack(fill="x", **kw)
        tk.Checkbutton(p, text=" Enable HOG human detection",
                       variable=self._hog_enabled, **ckw).pack(fill="x", **kw)

    # ═════════════════════════════════════════════════════════════════
    # HELPERS
    # ═════════════════════════════════════════════════════════════════

    def _btn(self, p, text, cmd, bg, fg):
        return tk.Button(p, text=text, command=cmd, bg=bg, fg=fg,
                         relief="flat", font=("Courier New", 8, "bold"),
                         padx=10, pady=4, cursor="hand2",
                         activebackground=ACC, activeforeground="#000")

    def _update_stats(self):
        self._stat["alerts"].config(text=str(self._alerts))
        self._stat["humans"].config(text=str(self._humans))
        self._stat["vehicles"].config(text=str(self._vehicles))
        self._stat["clips"].config(text=str(self._clips))

    # ═════════════════════════════════════════════════════════════════
    # CONTROLS
    # ═════════════════════════════════════════════════════════════════

    def _start(self):
        if self._running: return
        self._running = True
        self._status_var.set("● STARTING...")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _stop(self):
        self._running = False
        if self.recorder: self.recorder.stop()
        self._status_var.set("● IDLE")
        self._fps_var.set("-- FPS")
        self.root.after(700, self._refresh_clips)

    def _load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        if path:
            self._stop(); time.sleep(0.5)
            self._source = path
            self._source_var.set(f"File: {os.path.basename(path)}")

    def _use_webcam(self):
        self._stop(); time.sleep(0.5)
        self._source = 0
        self._source_var.set("Webcam (index 0)")

    def _update_sensitivity(self, _=None):
        self.detector._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=int(self._sensitivity.get()),
            detectShadows=True)

    def _reset_bg(self):
        self.detector.reset()
        self._status_var.set("● BG RESET")

    def _refresh_clips(self):
        self.clips_list.delete(0, "end")
        clips = []
        if os.path.exists("output"):
            clips = sorted(
                [f for f in os.listdir("output")
                 if f.endswith((".avi", ".mp4"))], reverse=True)
        for c in clips:
            self.clips_list.insert("end", c)
        self._clips = len(clips)
        self._update_stats()

    def _play_clip(self):
        sel = self.clips_list.curselection()
        if not sel: return
        path = os.path.join("output", self.clips_list.get(sel[0]))
        if os.path.exists(path): os.startfile(path)

    # ═════════════════════════════════════════════════════════════════
    # POLLING  (main thread — 100% thread safe)
    # ═════════════════════════════════════════════════════════════════

    def _poll_frame(self):
        """Called by root.after every 33ms — displays latest frame."""
        try:
            frame_data = None
            # Drain queue, keep only latest
            while True:
                frame_data = self._frame_queue.get_nowait()
        except queue.Empty:
            pass

        if frame_data is not None:
            try:
                rgb   = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                img   = Image.fromarray(rgb)
                w = self.feed_label.winfo_width()
                h = self.feed_label.winfo_height()
                if w < 50: w = 820
                if h < 50: h = 480
                img   = img.resize((w, h), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self._imgtk = imgtk          # keep reference!
                self.feed_label.config(image=imgtk, text="")
                self.feed_label.image = imgtk
            except Exception as e:
                print(f"[DISPLAY] {e}")

        self.root.after(33, self._poll_frame)

    def _poll_alerts(self):
        """Called by root.after — processes queued alert messages."""
        try:
            while True:
                msg, top = self._alert_queue.get_nowait()
                self.alert_list.insert(0, msg)
                if self.alert_list.size() > 300:
                    self.alert_list.delete("end")
                self._alerts += 1
                if top == "HUMAN":   self._humans += 1
                elif top == "VEHICLE": self._vehicles += 1
                self._update_stats()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_alerts)

    # ═════════════════════════════════════════════════════════════════
    # PROCESSING LOOP  (background thread — never touches Tkinter)
    # ═════════════════════════════════════════════════════════════════

    def _run_loop(self):
        source = VideoSource(self._source)
        try:
            source.open()
        except RuntimeError as e:
            messagebox.showerror("Error", str(e))
            self._running = False
            self._status_var.set("● ERROR")
            return

        # Silent warmup
        if self._source == 0:
            self._status_var.set("● WARMING UP...")
            for _ in range(20):
                ret, wf = source.read()
                if ret:
                    wf = cv2.resize(wf, (FRAME_WIDTH, FRAME_HEIGHT))
                    self.detector.process(wf)

        self._status_var.set("● LIVE")
        self.recorder = IncidentRecorder(source.fps, FRAME_WIDTH, FRAME_HEIGHT)
        prev_time = time.time()
        last_alert_ts = ""

        while self._running:
            ret, frame = source.read()
            if not ret:
                self._status_var.set("● VIDEO ENDED")
                break

            frame  = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            boxes  = self.detector.process(frame)
            boxes  = self._merge_boxes(boxes)
            labels = (self.classifier.classify(frame, boxes)
                      if self._hog_enabled.get() and boxes
                      else ["UNKNOWN"] * len(boxes))
            self.tracker.update(boxes, labels)
            tracked  = self.tracker.get_tracked_objects()
            motion   = self.detector.is_motion

            display = frame.copy()
            if self._show_mask.get():
                display = self.annotator.draw_fg_mask_overlay(
                    display, self.detector.get_foreground_mask())
            display = self.annotator.draw_tracked_objects(display, tracked)
            if motion:
                display = self.annotator.draw_alert_flash(display)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 0.001)
            prev_time = now
            display = self.annotator.draw_status_bar(display, fps, motion, len(tracked))
            self._fps_var.set(f"{fps:.0f} FPS")

            # Push frame to queue (drop if full — never block)
            try:
                self._frame_queue.put_nowait(display)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(display)
                except Exception:
                    pass

            # Recording
            top_label = labels[0] if labels else "UNKNOWN"
            was_rec   = self.recorder._recording
            self.recorder.feed(frame, motion, top_label)
            if not was_rec and self.recorder._recording:
                self._clips += 1

            # Alert (max 1 per second)
            if motion and tracked:
                ts = datetime.now().strftime("%H:%M:%S")
                if ts != last_alert_ts:
                    last_alert_ts = ts
                    msg = f"{ts}  {top_label}  [{len(tracked)} obj]"
                    try:
                        self._alert_queue.put_nowait((msg, top_label))
                    except queue.Full:
                        pass

        source.release()
        self._running = False
        self._status_var.set("● STOPPED")
        self.root.after(300, self._refresh_clips)

    def _merge_boxes(self, boxes):
        if len(boxes) < 2: return boxes
        rects = [[x, y, x+w, y+h] for x, y, w, h in boxes]
        used  = [False] * len(rects)
        out   = []
        for i in range(len(rects)):
            if used[i]: continue
            x1, y1, x2, y2 = rects[i]
            for j in range(i+1, len(rects)):
                if used[j]: continue
                if (min(x2, rects[j][2]) > max(x1, rects[j][0]) and
                        min(y2, rects[j][3]) > max(y1, rects[j][1])):
                    x1=min(x1,rects[j][0]); y1=min(y1,rects[j][1])
                    x2=max(x2,rects[j][2]); y2=max(y2,rects[j][3])
                    used[j] = True
            out.append((x1, y1, x2-x1, y2-y1))
            used[i] = True
        return out