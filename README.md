link dataset : https://www.kaggle.com/datasets/jumpingdino/german-credit-dataset

📊 Tổng quan Dữ liệu
Kích thước: 1,000 bản ghi và 21 đặc trưng (features).

Đặc trưng chính:

Thông tin tài chính: Số tiền vay (credit_amount), thời hạn vay (month_duration), lịch sử tín dụng.

Thông tin cá nhân: Tuổi, tình trạng việc làm, mục đích vay, nhà ở.

Biến mục tiêu: target (0: Bad, 1: Good).

🛠 Quy trình Xử lý (Workflow)
Phân tích dữ liệu (EDA): Trực quan hóa phân phối tuổi, số tiền vay và mối quan hệ giữa thời hạn vay với rủi ro.

Tiền xử lý:

Mã hóa biến mục tiêu sang dạng nhị phân.

Sử dụng One-hot Encoding cho các biến phân loại.

Chuẩn hóa dữ liệu bằng StandardScaler để đưa các biến số về cùng một thang đo.

Huấn luyện mô hình: So sánh giữa Logistic Regression và Random Forest.

📈 Kết quả đạt được
Mô hình tốt nhất: Logistic Regression đạt độ chính xác (Accuracy) khoảng 80.5%.

Nhận định quan trọng:

Số tiền vay lớn và thời hạn vay dài tỉ lệ thuận với rủi ro nợ xấu.

Độ tuổi càng cao có xu hướng là nhóm khách hàng vay an toàn hơn.

Mô hình Logistic Regression (tuyến tính) hoạt động hiệu quả hơn Random Forest trên tập dữ liệu này.

⚠️ Thách thức & Hướng phát triển
Thách thức: Dữ liệu bị mất cân bằng lớp (số lượng khách hàng 'Good' nhiều hơn 'Bad'), khiến mô hình chưa thực sự nhạy bén trong việc phát hiện khách hàng rủi ro cao.

Hướng phát triển:

Áp dụng kỹ thuật SMOTE để cân bằng dữ liệu.

Thử nghiệm các thuật toán mạnh hơn như XGBoost hoặc LightGBM.

Điều chỉnh trọng số lớp (class_weight='balanced') để cải thiện chỉ số Recall cho nhóm khách hàng xấu.
🚀 Cài đặt và Sử dụng

pip install pandas seaborn matplotlib scikit-learn
python finance.py
