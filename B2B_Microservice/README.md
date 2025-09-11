# ParkWave Demo (mini)

Yêu cầu: Docker & docker-compose.

1) Copy `.env.sample` → `.env` và chỉnh SECRET nếu muốn.
2) Build & chạy:
   docker-compose up --build

3) Truy cập:
   - Admin UI (dashboard): http://localhost:8000/
   - Kiosk (public): http://localhost:8000/site/site_demo_1
   - Payment Sandbox (manual simulate): http://localhost:9000/

Flow demo:
 - cv-worker (mô phỏng) sẽ gửi slot_update events mỗi vài giây.
 - Admin UI hiển thị spots & sessions; bạn có thể click "Set Occ" / "Set Free".
 - Khi có session và billed_amount unset, bạn có thể gọi API /api/payments/create?session_id=...&amount=... (ví dụ dùng curl)
 - Nối payment_url trả về (ví dụ http://localhost:9000/pay?payment_id=... ) và nhấn Pay (Success) → payment sandbox sẽ POST webhook tới backend (HMAC).
 - Backend xử lí webhook và tạo invoice; Admin -> Export invoices to download CSV.

Ghi chú:
 - cv-worker hiện đang mô phỏng. Để thay bằng OpenCV: chỉnh file cv-worker/worker.py theo logic đề xuất trong báo cáo.
 - DB là Postgres container; nếu muốn dùng sqlite, thay DATABASE_URL trong .env.