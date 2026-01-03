import sys, os, time
import numpy as np
import cv2
from ultralytics import YOLO
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QGraphicsView, QGraphicsScene,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QMessageBox,
    QSizePolicy, QGroupBox, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QRectF
from PySide6.QtGui import QPixmap, QImage

# ====================== Color palette & helpers ======================
PALETTE = np.array([
    [255,56,56],[255,157,151],[255,112,31],[255,178,29],[207,210,49],[72,249,10],
    [146,204,23],[61,219,134],[26,147,52],[0,212,187],[44,153,168],[0,194,255],
    [52,69,147],[100,115,255],[0,24,236],[132,56,255],[82,0,133],[203,56,255],
    [255,149,200],[255,55,199]
], dtype=np.uint8)

def class_color(cid): return tuple(int(x) for x in PALETTE[cid % len(PALETTE)])

def rect_intersect(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

def in_bounds(r, W, H):
    return r[0] >= 0 and r[1] >= 0 and r[2] <= W and r[3] <= H

# ---------- Build a horizontal label patch, optionally rotate 90° CCW ----------
def make_label_patch(text, box_color, fg=(255,255,255),
                     font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.9, thickness=2,
                     pad=4, alpha=0.55, rotate90=False):
    (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
    w = tw + 2*pad
    h = th + base + 2*pad
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(patch, (0, 0), (w-1, h-1), box_color, -1)
    cv2.putText(patch, text, (pad, pad + th), font, font_scale, fg, thickness, cv2.LINE_AA)
    if rotate90:
        patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return patch, alpha

def paste_patch(img, patch, x, y, alpha=0.55):
    H, W = img.shape[:2]
    ph, pw = patch.shape[:2]
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(W, x + pw); y2 = min(H, y + ph)
    if x1 >= x2 or y1 >= y2:
        return None
    px1, py1 = x1 - x, y1 - y
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
    roi = img[y1:y2, x1:x2]
    sub = patch[py1:py2, px1:px2]
    blended = cv2.addWeighted(sub, alpha, roi, 1 - alpha, 0)
    img[y1:y2, x1:x2] = blended
    return (x1, y1, x2, y2)

def place_label_nonoverlap(img, cx, top_y, bot_y, text, box_color, occupied,
                           font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.9, thick_text=2,
                           pad=4, margin_y=8, alpha=0.55,
                           grid_step=8, max_tries=80, rotate90=False):
    patch, a = make_label_patch(text, box_color, font=font, font_scale=font_scale,
                                thickness=thick_text, pad=pad, alpha=alpha, rotate90=rotate90)
    ph, pw = patch.shape[:2]
    H, W = img.shape[:2]

    def rect_from_center(cx, baseline_y):
        tx = int(cx - pw / 2)
        ty = int(baseline_y - ph)
        tx = max(2, min(tx, W - pw - 2))
        return (tx, ty, tx + pw, ty + ph), (tx, ty)

    candidates = [
        bot_y + margin_y + ph,
        top_y - margin_y,
        min(bot_y - 6, H - ph - 2),
        max(top_y + 6, 2 + ph)
    ]
    for baseline in candidates:
        rect, (tx, ty) = rect_from_center(cx, baseline)
        if rect[1] < 0 or rect[3] > H:
            continue
        if any(rect_intersect(rect, occ) for occ in occupied):
            continue
        res = paste_patch(img, patch, tx, ty, a)
        if res is not None:
            return res

    offsets = [0]
    for k in range(1, max_tries + 1):
        offsets += [k*grid_step, -k*grid_step]
    start = bot_y + margin_y + ph
    for dy in offsets:
        b = start + dy
        rect, (tx, ty) = rect_from_center(cx, b)
        if not in_bounds(rect, W, H): continue
        if any(rect_intersect(rect, occ) for occ in occupied): continue
        res = paste_patch(img, patch, tx, ty, a)
        if res is not None:
            return res

    b = min(bot_y - 6, H - ph - 2)
    rect, (tx, ty) = rect_from_center(cx, b)
    res = paste_patch(img, patch, tx, ty, a)
    return rect if res is None else res

# ====================== Inference thread ======================
class InferWorker(QThread):
    finished = Signal(dict)
    def __init__(self, model: YOLO, img_path: str, conf: float, iou: float, imgsz: int):
        super().__init__()
        self.model, self.img_path, self.conf, self.iou, self.imgsz = model, img_path, conf, iou, imgsz
    def run(self):
        t0 = time.time()
        results = self.model.predict(source=self.img_path, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
        r = results[0]
        use_obb = hasattr(r, "obb") and (r.obb is not None)
        if use_obb:
            boxes_xyxy = r.obb.xyxy.cpu().numpy()
            polys      = r.obb.xyxyxyxy.cpu().numpy()
            xywhr      = r.obb.xywhr.cpu().numpy()
            clses      = r.obb.cls.int().cpu().numpy()
            confs      = r.obb.conf.cpu().numpy()
        else:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            polys      = None
            xywhr      = None
            clses      = r.boxes.cls.int().cpu().numpy()
            confs      = r.boxes.conf.cpu().numpy()
        names = r.names
        img_bgr = cv2.imread(self.img_path)
        out = dict(
            img=img_bgr, names=names, use_obb=use_obb,
            boxes=boxes_xyxy, polys=polys, xywhr=xywhr, clses=clses, confs=confs,
            time_ms=1000*(time.time()-t0)
        )
        self.finished.emit(out)

# ====================== Image view ======================
class ImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._pixmap_item = None
        self._rgb_backing = None
    def set_image(self, bgr_img: np.ndarray):
        if bgr_img is None: return
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        self._rgb_backing = np.ascontiguousarray(rgb)
        h, w, ch = self._rgb_backing.shape
        qimg = QImage(self._rgb_backing.data, w, h, ch*w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.scene().clear()
        self._pixmap_item = self.scene().addPixmap(pix)
        self.setSceneRect(QRectF(0, 0, w, h))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
    def wheelEvent(self, e):
        factor = 1.15 if e.angleDelta().y() > 0 else 1/1.15
        self.scale(factor, factor)

# ====================== Main window ======================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Overhead Conductor Compression Fitting X-ray Defect Detection — Visualization & Assessment")
        self.model = None
        self.img_path = ""

        # ---- Light theme stylesheet (cards / inputs / tables / buttons) ----
        self.setStyleSheet("""
            QMainWindow { background: #f5f7fb; }
            QLabel { color: #1f2937; font-size: 13px; }
            QGroupBox {
                background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;
                margin-top: 10px; padding: 10px 12px 12px 12px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px;
                color:#111827; font-weight: 600; background: transparent; }

            QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox {
                background:#ffffff; border:1px solid #d1d5db; border-radius:6px; padding:4px 6px;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {
                border:1px solid #60a5fa;
            }

            /* replace your QPushButton styles with this */
            QPushButton {
                background:#E8F0FF;      /* very light blue */
                color:#0F172A;           /* 深色文字 */
                border:1px solid #D9E6FF;
                border-radius:8px;
                padding:8px 10px;
                font-weight:600;
            }
            QPushButton:hover   { background:#DFEAFF; }  /* 略深一点点做悬停 */
            QPushButton:pressed { background:#D0DFFF; }  /* 按下 */
            QPushButton:disabled{
                background:#F4F8FF;
                color:#9AA4B2;
                border:1px solid #EEF3FF;
            }


            QTableWidget {
                background:#ffffff; border:1px solid #e5e7eb; border-radius:8px;
                gridline-color:#e5e7eb; alternate-background-color:#fafbff;
            }
            QHeaderView::section {
                background:#f3f4f6; color:#111827; padding:6px; border: 0px; border-right:1px solid #e5e7eb;
            }
            QTableWidget::item { padding:6px; }
            QScrollBar:vertical { background:#f3f4f6; width:10px; border:none; }
            QScrollBar::handle:vertical { background:#d1d5db; border-radius:5px; }
            QStatusBar { background:#ffffff; border-top:1px solid #e5e7eb; }
        """)

        # ---------- left panel widgets ----------
        self.ed_model = QLineEdit("/home/zgc/datawork/DRimage/ultralytics-main0924/ultralytics-main/datas/o_crimp/runs/obb/train10/weights/best.pt")
        self.btn_load = QPushButton("Load Model")
        self.ed_img   = QLineEdit("/home/zgc/datawork/DRimage/ultralytics-main0924/ultralytics-main/datas/o_crimp/images/test/uc_0242.png")
        self.btn_img  = QPushButton("Select Image")
        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0,1); self.spin_conf.setValue(0.5)
        self.spin_iou  = QDoubleSpinBox(); self.spin_iou.setRange(0,1); self.spin_iou.setValue(0.45)
        self.spin_imgsz= QSpinBox();       self.spin_imgsz.setRange(256,4096); self.spin_imgsz.setValue(1024)
        self.chk_labels= QCheckBox("Show labels"); self.chk_labels.setChecked(True)
        self.spin_alpha= QDoubleSpinBox(); self.spin_alpha.setRange(0,1); self.spin_alpha.setValue(0.65)
        self.spin_font = QDoubleSpinBox(); self.spin_font.setRange(0.4,2.5); self.spin_font.setValue(1.0)
        self.spin_thk  = QSpinBox();       self.spin_thk.setRange(1,6); self.spin_thk.setValue(2)
        self.spin_step = QSpinBox();       self.spin_step.setRange(2,40); self.spin_step.setValue(11)
        self.spin_try  = QSpinBox();       self.spin_try.setRange(10,300); self.spin_try.setValue(83)
        self.btn_run   = QPushButton("Run Inference")
        self.btn_save  = QPushButton("Save Render PNG")

        def row(widget, label):
            w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0,0,0,8)
            h.addWidget(QLabel(label)); h.addWidget(widget); return w

        # group: Model
        gb_model = QGroupBox("Model")
        l_model = QVBoxLayout(gb_model)
        l_model.addWidget(row(self.ed_model,"Model path"))
        l_model.addWidget(self.btn_load)

        # group: Image
        gb_image = QGroupBox("Image")
        l_image = QVBoxLayout(gb_image)
        l_image.addWidget(row(self.ed_img,"Image path"))
        l_image.addWidget(self.btn_img)

        # group: Inference Params
        gb_params = QGroupBox("Inference Params")
        l_params = QVBoxLayout(gb_params)
        l_params.addWidget(row(self.spin_conf,"conf"))
        l_params.addWidget(row(self.spin_iou,"iou"))
        l_params.addWidget(row(self.spin_imgsz,"imgsz"))

        # group: Rendering params
        gb_render = QGroupBox("Rendering")
        l_render = QVBoxLayout(gb_render)
        l_render.addWidget(self.chk_labels)
        l_render.addWidget(row(self.spin_alpha,"alpha"))
        l_render.addWidget(row(self.spin_font,"Font scale"))
        l_render.addWidget(row(self.spin_thk,"Box/Text thickness"))
        # Optional hidden advanced:
        adv = QWidget(); adv_l = QVBoxLayout(adv); adv_l.setContentsMargins(0,0,0,0)
        adv_l.addWidget(row(self.spin_step,"Label offset step"))
        adv_l.addWidget(row(self.spin_try,"Offset tries"))
        adv.setVisible(False)  # keep logic but hide by default
        l_render.addWidget(adv)

        # group: Actions
        gb_actions = QGroupBox("Actions")
        l_act = QVBoxLayout(gb_actions); l_act.addWidget(self.btn_run); l_act.addWidget(self.btn_save)

        # left column container
        left = QWidget(); lyt_left = QVBoxLayout(left)
        lyt_left.addWidget(gb_model)
        lyt_left.addWidget(gb_image)
        lyt_left.addWidget(gb_params)
        lyt_left.addWidget(gb_render)
        lyt_left.addWidget(gb_actions)
        lyt_left.addStretch(1)

        # Center view (framed)
        self.view = ImageView()
        center_frame = QGroupBox("Preview")
        center_layout = QVBoxLayout(center_frame); center_layout.addWidget(self.view)

        # ===== Fixed right panel width =====
        self.RIGHT_PANEL_W = 520

        # Right tables (grouped)
        def build_table(headers):
            t = QTableWidget(0, len(headers))
            t.setHorizontalHeaderLabels(headers)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
            t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            t.setWordWrap(True)
            t.setTextElideMode(Qt.ElideNone)
            t.verticalHeader().setVisible(False)
            t.setAlternatingRowColors(True)
            return t

        self.defect_table = build_table(["#", "class_id", "class", "conf", "xc,yc,w,h,angle"])
        self.aux_table    = build_table(["#", "class_id", "class", "conf", "xc,yc,w,h,angle"])
        self.eval_table   = build_table(["Description", "Result", "Severity"])
        self.overall_table = QTableWidget(0, 1)
        self.overall_table.horizontalHeader().setVisible(False)
        self.overall_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.overall_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.overall_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.overall_table.setWordWrap(True)
        self.overall_table.setTextElideMode(Qt.ElideNone)
        self.overall_table.verticalHeader().setVisible(False)
        self.overall_table.setAlternatingRowColors(True)

        gb_defect = QGroupBox("Defect targets");  ld = QVBoxLayout(gb_defect);  ld.addWidget(self.defect_table)
        gb_aux    = QGroupBox("Auxiliary targets"); la = QVBoxLayout(gb_aux);    la.addWidget(self.aux_table)
        gb_eval   = QGroupBox("Defect assessment"); le = QVBoxLayout(gb_eval);   le.addWidget(self.eval_table)
        gb_over   = QGroupBox("Overall");           lo = QVBoxLayout(gb_over);   lo.addWidget(self.overall_table)

        # Layout composition
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(center_frame)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(gb_defect)
        right_layout.addWidget(gb_aux)
        right_layout.addWidget(gb_eval)
        right_layout.addWidget(gb_over)
        right_panel.setFixedWidth(self.RIGHT_PANEL_W)
        splitter.addWidget(right_panel)

        splitter.setSizes([300, 820, self.RIGHT_PANEL_W])
        self.setCentralWidget(splitter)

        # Status bar
        self.status_label = QLabel("Status: Ready")
        self.statusBar().addPermanentWidget(self.status_label)

        # Signals
        self.btn_load.clicked.connect(self.load_model)
        self.btn_img.clicked.connect(self.pick_image)
        self.btn_run.clicked.connect(self.run_infer)
        self.btn_save.clicked.connect(self.save_png)

        self._apply_table_layouts()

    # ---------- Table helpers ----------
    def _equalize_columns(self, table: QTableWidget, weights=None):
        n = table.columnCount()
        if n <= 0:
            return
        viewport_w = table.viewport().width()
        if viewport_w <= 0:
            viewport_w = self.RIGHT_PANEL_W - 20

        min_w = 40
        if weights and len(weights) == n:
            total = float(sum(weights))
            widths = [max(min_w, int(viewport_w * (w / total))) for w in weights]
            used = sum(widths[:-1])
            widths[-1] = max(min_w, viewport_w - used)
            for c, w in enumerate(widths):
                table.setColumnWidth(c, w)
        else:
            base = max(min_w, viewport_w // n)
            for c in range(n - 1):
                table.setColumnWidth(c, base)
            table.setColumnWidth(n - 1, max(min_w, viewport_w - base * (n - 1)))

    def _apply_table_layouts(self):
        weights_5cols = [0.02, 0.12, 0.20, 0.12, 0.52]
        self._equalize_columns(self.defect_table, weights_5cols)
        self._equalize_columns(self.aux_table,    weights_5cols)
        self._equalize_columns(self.eval_table, None)

        for t in (self.defect_table, self.aux_table, self.eval_table, self.overall_table):
            t.resizeRowsToContents()

    def _set_center_item(self, table: QTableWidget, r: int, c: int, text: str):
        it = QTableWidgetItem(text)
        it.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        table.setItem(r, c, it)

    def _wrapable(self, s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        return s.replace(",", ",\u200B").replace("，", "，\u200B")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_table_layouts()

    # ---------- App logic ----------
    def load_model(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Model Files (*.pt)")
        if model_file:
            self.ed_model.setText(model_file)
            self.model = YOLO(model_file)
            self.status_label.setText("Status: Model loaded")

    def pick_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", os.path.dirname(self.ed_img.text()), "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.ed_img.setText(path)
            img = cv2.imread(path)
            self.view.set_image(img)

    def run_infer(self):
        if self.model is None:
            QMessageBox.information(self, "Info", "Please load a model first."); return
        self.img_path = self.ed_img.text().strip()
        if not os.path.exists(self.img_path):
            QMessageBox.warning(self, "Error", f"Image not found:\n{self.img_path}"); return
        self.worker = InferWorker(self.model, self.img_path, self.spin_conf.value(), self.spin_iou.value(), self.spin_imgsz.value())
        self.worker.finished.connect(self.on_infer_done)
        self.status_label.setText("Status: Inferring…")
        self.worker.start()

    # Severity helpers (English labels)
    def _sev_val(self, s:str)->int: return {"Pass":0, "Minor defect":1, "Major defect":2, "Critical defect":3}.get(s,0)
    def _sev_merge(self, a:str, b:str)->str:
        order = ["Pass","Minor defect","Major defect","Critical defect"]
        return order[max(self._sev_val(a), self._sev_val(b))]

    def on_infer_done(self, out: dict):
        try:
            img = out['img'].copy()
            names = out['names']
            use_obb = out['use_obb']
            boxes, polys, xywhr, clses, confs = out['boxes'], out['polys'], out['xywhr'], out['clses'], out['confs']

            TH = int(self.spin_thk.value())
            font_scale = float(self.spin_font.value())
            alpha = float(self.spin_alpha.value())
            grid_step = int(self.spin_step.value())
            max_tries = int(self.spin_try.value())
            show_labels = self.chk_labels.isChecked()

            idx = np.arange(len(clses))
            if len(idx) == 0:
                self.view.set_image(img)
                for t in (self.defect_table, self.aux_table, self.eval_table, self.overall_table):
                    t.clearContents(); t.setRowCount(0)
                self._apply_table_layouts()
                self.status_label.setText(f"Status: Done (no objects), {out['time_ms']:.1f} ms")
                return

            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            order = idx[np.argsort(-areas)]
            occupied = []

            for i in order:
                cid = int(clses[i]); conf = float(confs[i])
                color = class_color(cid)
                x1, y1, x2, y2 = boxes[i]
                if use_obb and polys is not None:
                    poly_pts = polys[i].reshape(-1,2).astype(int)
                    cv2.polylines(img, [poly_pts], True, color, TH, lineType=cv2.LINE_AA)
                else:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, TH, lineType=cv2.LINE_AA)
                if show_labels:
                    cx = (x1 + x2) / 2
                    label = f"{names[cid]} {conf:.2f}"
                    rotate90 = names[cid] in ["P_crimping", "M_crimping", "Is_crimping"]
                    rect = place_label_nonoverlap(img, cx, y1, y2, label, color, occupied,
                                                  font_scale=font_scale, thick_text=TH, alpha=alpha,
                                                  grid_step=grid_step, max_tries=max_tries, rotate90=rotate90)
                    occupied.append(rect)

            self.view.set_image(img)
            self.status_label.setText(f"Status: Done, {out['time_ms']:.1f} ms")

            # Fill tables
            self.defect_table.clearContents(); self.defect_table.setRowCount(0)
            self.aux_table.clearContents();    self.aux_table.setRowCount(0)
            defect_rows, aux_rows = [], []
            for k, i in enumerate(order):
                cid = int(clses[i]); conf = float(confs[i])
                if use_obb and xywhr is not None:
                    xywhr_i = xywhr[i].reshape(-1)
                    xywhr_str = ",".join([f"{v:.1f}" for v in xywhr_i])
                else:
                    xywhr_str = "-"
                if cid in (3, 4):
                    aux_rows.append((k, cid, names[cid], f"{conf:.2f}", xywhr_str))
                else:
                    defect_rows.append((k, cid, names[cid], f"{conf:.2f}", xywhr_str))

            for tbl, rows in ((self.defect_table, defect_rows), (self.aux_table, aux_rows)):
                tbl.setRowCount(len(rows))
                for r,(idx_show, cid, cls_name, conf_s, xywhr_s) in enumerate(rows):
                    self._set_center_item(tbl, r, 0, str(idx_show))
                    self._set_center_item(tbl, r, 1, str(cid))
                    self._set_center_item(tbl, r, 2, cls_name)
                    self._set_center_item(tbl, r, 3, conf_s)
                    self._set_center_item(tbl, r, 4, self._wrapable(xywhr_s))

            self._apply_table_layouts()

            # Assessment
            rows = []
            overall = "Pass"

            has_fracture = any(int(c)==5 for c in clses)
            if has_fracture:
                level = "Critical defect"
                overall = self._sev_merge(overall, level)
                rows.append(("Steel-core fracture", "Fail", level))

            cnt_M  = sum(int(c)==2 for c in clses)
            cnt_Is = sum(int(c)==1 for c in clses)
            cnt_P  = sum(int(c)==3 for c in clses)
            total_slots  = cnt_M + cnt_Is + cnt_P
            defect_slots = cnt_M + cnt_Is
            if total_slots>0 and defect_slots>0:
                N_ratio = defect_slots/total_slots*100.0
                level = "Minor defect" if 0 < N_ratio < 50 else "Major defect"
                parts = []
                if cnt_M>0:  parts.append(f"Under-crimp {cnt_M} serrations")
                if cnt_Is>0: parts.append(f"Insufficient-crimp {cnt_Is} serrations")
                desc = ", ".join(parts) + f", defect serration ratio {N_ratio:.0f}%"
                overall = self._sev_merge(overall, level)
                rows.append((desc, "Fail", level))

            if use_obb and (xywhr is not None):
                cavity_wh = [max(float(w), float(h)) for (_,_,w,h,_), c in zip(xywhr, clses) if int(c)==0]
                anchor_wh = [max(float(w), float(h)) for (_,_,w,h,_), c in zip(xywhr, clses) if int(c)==4]
                if len(cavity_wh)>0 and len(anchor_wh)>0 and max(anchor_wh)>0:
                    C_l = max(cavity_wh); S_l = max(anchor_wh)
                    M_ratio = C_l / S_l * 100.0
                    if M_ratio < 10:
                        level = "Minor defect"
                    elif M_ratio < 30:
                        level = "Major defect"
                    else:
                        level = "Critical defect"
                    desc = f"Insufficient steel-core insertion, cavity ratio {M_ratio:.0f}%"
                    overall = self._sev_merge(overall, level)
                    rows.append((desc, "Fail", level))

            self.eval_table.clearContents()
            if not rows:
                self.eval_table.setRowCount(1)
                self._set_center_item(self.eval_table, 0, 0, "—")
                self._set_center_item(self.eval_table, 0, 1, "Pass")
                self._set_center_item(self.eval_table, 0, 2, "Pass")
            else:
                self.eval_table.setRowCount(len(rows))
                for r,(desc,res,level) in enumerate(rows):
                    self._set_center_item(self.eval_table, r, 0, desc)
                    self._set_center_item(self.eval_table, r, 1, res)
                    self._set_center_item(self.eval_table, r, 2, level)

            self._apply_table_layouts()

            # Overall verdict
            def overall_text(final_level:str)->str:
                if final_level == "Minor defect":
                    return "Overall result: Fail — Minor defect. Remedy during next maintenance."
                if final_level == "Major defect":
                    return "Overall result: Fail — Major defect. Potential safety risk; fix within one week."
                if final_level == "Critical defect":
                    return "Overall result: Fail — Critical defect. Serious safety hazard; fix immediately."
                return "Overall result: Pass."

            self.overall_table.clearContents()
            self.overall_table.setRowCount(1)
            self.overall_table.setItem(0, 0, QTableWidgetItem(overall_text(overall)))

            self._apply_table_layouts()
            self._last_render = img

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Status: Error: {e}")

    def save_png(self):
        if not hasattr(self, "_last_render"):
            QMessageBox.information(self, "Info", "No rendered result yet."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "vis.png", "PNG Images (*.png)")
        if path:
            cv2.imwrite(path, self._last_render)
            self.status_label.setText(f"Status: Saved {path}")

# ====================== Entry ======================
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1500, 880)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
