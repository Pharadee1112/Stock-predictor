# IMPLEMENT.md — แผนงานเพื่อทำให้ Stock Predictor ใช้งานได้จริง

เอกสารนี้เป็น checklist งานแบ่งเป็น phase สำหรับพัฒนาโปรเจกต์ต่อจากสถานะปัจจุบัน (ทำนายราคาหุ้นด้วย ML จาก tvDatafeed + หน้าเว็บ Flask)
เช็คบ็อกซ์แต่ละอันเมื่อทำเสร็จ เพื่อไม่ให้หลงระหว่างทาง

**ลำดับที่แนะนำ: ทำ Phase 1 ให้ครบก่อนเสมอ** — เพราะกระทบความน่าเชื่อถือของ "การทำนาย" ซึ่งเป็นแก่นของโปรเจกต์ ส่วน Phase 2-4 ทำตามลำดับหรือสลับได้ตามความจำเป็น

---

## Phase 1 — Methodology (ความแม่นยำ/ความน่าเชื่อถือของโมเดล)

เป้าหมาย: ตัวเลขและกราฟที่โชว์ผู้ใช้ต้องสะท้อนความสามารถจริงของโมเดล ไม่ใช่ training error ที่หลอกตา

- [x] **Train/test split แบบ chronological**
  ไฟล์: `stock_analyzer.py` (`predict_future_close`)
  แบ่งข้อมูล 80/20 ตามลำดับเวลา (ห้ามสุ่ม) เทรนด้วย train, วัด MAE/MSE/MAPE จาก test เท่านั้น
  ผลลัพธ์: ตัวเลข error ที่ส่งกลับ frontend ต้องเป็น out-of-sample error

- [x] **แก้ LSTM ให้ใช้ sliding window ของราคาจริง**
  ไฟล์: `stock_analyzer.py` (`model_type == 'lstm'` branch)
  สร้าง X จากราคาปิดย้อนหลัง N วัน (เช่น window=20) แทน day-index
  ทำ multi-step rollout (ใช้ค่าที่ทำนายได้ต่อยอดทำนายวันถัดไป) สำหรับ `days_ahead > 1`

- [x] **เพิ่ม baseline model เทียบ**
  ใช้ naive forecast ("พรุ่งนี้ = ราคาวันนี้" หรือ moving average ง่ายๆ) เป็นตัวเทียบ
  ถ้าโมเดลที่เลือกไม่ดีกว่า baseline → ควรมี flag/คำเตือนแจ้งผู้ใช้

- [x] **เพิ่ม feature อื่นนอกจาก day-index**
  เช่น moving average (MA5/MA20), volume, volatility (rolling std)
  จำเป็นโดยเฉพาะกับโมเดล tree-based (RandomForest/GradientBoosting) ที่ทำงานได้แย่กับ feature เดียวที่เป็น index ไล่ขึ้นตรงๆ

- [x] **แสดงระดับความไม่แน่นอนของการทำนาย**
  เช่น error band จาก residual std หรือข้อความเตือนที่ปรับตาม `days_ahead` ("ทำนายไกลกว่า 7 วัน ความแม่นยำจะลดลงมาก")

- [x] **เพิ่ม MAPE (% error) ควบคู่ MAE/MSE**
  อ่านง่ายกว่าสำหรับผู้ใช้ทั่วไปที่ไม่ใช่สายเทคนิค

---

## Phase 2 — Error Handling & Validation (กันแอปพัง)

เป้าหมาย: input ที่ไม่สมบูรณ์แบบ (ผู้ใช้จริงพิมพ์ผิดบ่อย) ต้องไม่ทำให้แอป 500 หรือค้าง

- [x] validate `date_input`: รูปแบบถูกต้อง และต้องเป็นวันที่ **หลัง** `last_date` เท่านั้น (reject ด้วย 400 + ข้อความชัดเจนถ้าไม่ผ่าน)
- [x] validate `data_points`: ต้องเป็นตัวเลขในช่วงที่กำหนด (เช่น 30–500) — window ของ LSTM ต้องการข้อมูลขั้นต่ำถึงจะมีความหมาย
- [x] validate `symbol`: ไม่ว่าง ไม่มีอักขระผิดปกติ
- [x] try/except รอบ `analyzer.collect_data(...)` ใน `app.py` — เช็คถ้า data เป็น `None`/ว่าง ให้ตอบ error message (HTTP 502) แทนปล่อยให้ throw
- [x] client-side validation ใน `templates/index.html` ก่อนยิง fetch (required fields, date ต้อง >= พรุ่งนี้)
- [x] timeout ให้ tvDatafeed call เผื่อ network ช้า/ค้าง

---

## Phase 3 — UX & Performance

เป้าหมาย: ใช้งานได้ลื่นขึ้น ไม่มีไฟล์ขยะสะสม ไม่ต้องเทรนซ้ำโดยไม่จำเป็น

- [x] เพิ่ม loading indicator ระหว่างรอผลลัพธ์ (`templates/index.html`) — LSTM/MLP ใช้เวลาหลายวินาที ตอนนี้ปุ่มไม่มี feedback ใดๆ
- [x] ส่งกราฟเป็น base64 ใน JSON response แทนการเซฟไฟล์ลง `static/` — แก้ปัญหาไฟล์ png สะสมไม่จำกัด (ปัจจุบันมีไฟล์ทดสอบค้างอยู่แล้วหลายสิบไฟล์ใน `static/`)
- [x] cache ผลลัพธ์/โมเดลต่อ (symbol, data_points, model_type) ช่วงสั้นๆ (เช่น 15 นาที) กันเทรนซ้ำทุก request เดิม
- [x] รองรับเลือก exchange แทน hardcode `NASDAQ` ใน `app.py`

---

## Phase 4 — ส่วนเสริม & เตรียม Deploy

เป้าหมาย: ต่อยอดให้ดูสมบูรณ์ขึ้นและพร้อมใช้งานนอกเครื่อง dev

- [ ] เพิ่มเลเยอร์ LLM สำหรับสรุปผลเป็นภาษาคน (อธิบาย prediction + error metrics ให้เข้าใจง่าย)
      **หมายเหตุ:** ใช้ LLM เป็นตัวอธิบายผลเท่านั้น ห้ามใช้ทำนายราคาตัวเลขโดยตรง (LLM ไม่มีความสามารถด้าน numeric forecasting)
- [x] เปลี่ยนจาก `app.run(debug=True)` เป็น WSGI server จริง (gunicorn/waitress) ตอน deploy
- [x] แยก config dev/prod ผ่าน `.env`
- [x] เขียน unit test พื้นฐานสำหรับ `StockAnalyzer` (mock tvDatafeed กันเทสต์พังเวลา API ล่มจริง)

---

## Progress Log

| Phase | สถานะ | หมายเหตุ |
|---|---|---|
| 1 — Methodology | เสร็จแล้ว | rewrite `predict_future_close` ทั้งหมด (classical + LSTM แยก path), แก้ `app.py`/`index.html` ให้ส่ง/แสดงผลลัพธ์ใหม่ครบ |
| 2 — Error Handling | เสร็จแล้ว | validate symbol/date/model_type/data_points + collect_data timeout (ThreadPoolExecutor) ใน `app.py`, client-side check ใน `index.html` |
| 3 — UX & Performance | เสร็จแล้ว | แยก `fit`/`rollout` ใน `stock_analyzer.py` เพื่อรองรับ cache, เพิ่ม in-memory TTL cache + exchange field + base64 plot + loading indicator ใน `app.py`/`index.html` |
| 4 — ส่วนเสริม/Deploy | บางส่วน | unit test (`tests/test_stock_analyzer.py`, 24 ผ่าน) + `.env`/`config.py` + waitress สำหรับ prod (`DEBUG=False` ใน `.env`) เสร็จแล้ว; เหลือแค่ LLM explainer layer (เลือก Anthropic ไว้แล้ว รอทำทีหลัง) |
