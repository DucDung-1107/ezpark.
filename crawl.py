import time
import csv
import urllib.parse
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
import json
import random
from datetime import datetime

# Define the Google Drive path
# Hãy đảm bảo bạn đã kết nối với Google Drive nếu chạy trên Colab
DRIVE_PATH = "/content/drive/MyDrive/ezpark"

# Ensure the directory exists
os.makedirs(DRIVE_PATH, exist_ok=True)

# Danh sách các thành phố/tỉnh lớn của Việt Nam
VIETNAM_CITIES = [
    "Hà Nội", "TP.HCM", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
    "Biên Hòa", "Huế", "Nha Trang", "Buôn Ma Thuột", "Quy Nhon",
    "Vũng Tàu", "Nam Định", "Phan Thiết", "Long Xuyên", "Hạ Long",
    "Thái Nguyên", "Thanh Hóa", "Rạch Giá", "Cam Ranh", "Vinh",
    "Mỹ Tho", "Đà Lạt", "Bến Tre", "Vĩnh Long", "Trà Vinh",
    "Sóc Trăng", "Bạc Liêu", "Cà Mau", "Tây Ninh", "An Giang",
    "Kiên Giang", "Bình Dương", "Đồng Nai", "Bà Rịa - Vũng Tàu",
    "Lâm Đồng", "Ninh Thuận", "Bình Thuận", "Kon Tum", "Gia Lai",
    "Đắk Lắk", "Đắk Nông", "Khánh Hòa", "Phú Yên", "Bình Định",
    "Quảng Ngãi", "Quảng Nam", "Thừa Thiên Huế", "Quảng Trị",
    "Quảng Bình", "Hà Tĩnh", "Nghệ An", "Thanh Hóa", "Ninh Bình",
    "Nam Định", "Thái Bình", "Hưng Yên", "Hà Nam", "Vĩnh Phúc",
    "Bắc Ninh", "Quảng Ninh", "Hải Dương", "Hòa Bình", "Sơn La"
]

def extract_lat_lng_from_url(url):
    try:
        if '@' in url:
            parts = url.split('@')[1].split(',')
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])

        import re
        pattern1 = r'!3d([0-9.-]+)!4d([0-9.-]+)'
        match1 = re.search(pattern1, url)
        if match1:
            return float(match1.group(1)), float(match1.group(2))

        pattern2 = r'/@([0-9.-]+),([0-9.-]+)'
        match2 = re.search(pattern2, url)
        if match2:
            return float(match2.group(1)), float(match2.group(2))

    except (ValueError, IndexError):
        pass
    return None, None

def extract_rating_from_aria_label(aria_label):
    try:
        import re
        match = re.search(r'(\d+\.?\d*)', aria_label)
        if match:
            rating_value = match.group(1)
            if rating_value and float(rating_value) > 0:
                return rating_value
    except:
        pass
    return ""

def get_reviews(driver, max_reviews=25):
    """
    Hàm được tối ưu để lấy reviews trực tiếp từ trang chi tiết của địa điểm.
    Nó sẽ cuộn trang để tải thêm reviews và sử dụng BeautifulSoup để trích xuất.
    """
    reviews = []
    try:
        # Cuộn trang để tải reviews
        print("    📜 Đang cuộn để tải reviews...")
        body_element = driver.find_element(By.TAG_NAME, 'body')
        # Cuộn nhiều lần để đảm bảo các review được tải
        for _ in range(10):
            # Gửi phím Page Down tới body để cuộn
            body_element.send_keys(webdriver.common.keys.Keys.PAGE_DOWN)
            time.sleep(0.3) # Đợi một chút để nội dung mới tải

        # Sử dụng BeautifulSoup để lấy nội dung reviews
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")

        # Selector đã được cập nhật để ổn định hơn
        # jJc9Ad: class của div chứa toàn bộ review
        # wiI7pd: class của span chứa nội dung review
        review_elements = soup.select('div.jJc9Ad span.wiI7pd, div.Jtu6Td span')

        for review_span in review_elements:
            text = review_span.get_text(strip=True)
            if text and len(text) > 15 and text not in reviews: # Lọc các review ngắn/nhiễu
                reviews.append(text)
                if len(reviews) >= max_reviews:
                    break

        if reviews:
            print(f"    📝 Tìm thấy {len(reviews)} reviews.")
        else:
            print("    📝 Không tìm thấy review nào.")

    except Exception as e:
        print(f"    ❌ Lỗi khi lấy reviews: {e}")

    return reviews

def save_results_immediately(results, query_name):
    """Lưu kết quả ngay lập tức sau mỗi query"""
    if not results:
        print("⚠️ Không có kết quả để lưu.")
        return None
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query_name = "".join(c for c in query_name if c.isalnum() or c in (' ', '-', '_')).rstrip()[:30]
        filename = f"{DRIVE_PATH}/query_{safe_query_name}_{timestamp}.csv"

        with open(filename, mode="w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["stt", "name", "address", "lat", "lng", "reviews", "rating", "query", "city"])
            writer.writeheader()
            for i, row in enumerate(results, 1):
                row_copy = row.copy()
                row_copy["stt"] = i
                row_copy["reviews"] = json.dumps(row["reviews"], ensure_ascii=False)
                writer.writerow(row_copy)

        print(f"💾 ĐÃ LƯU: {len(results)} kết quả -> {filename}")
        return filename
    except Exception as e:
        print(f"❌ Lỗi lưu file: {e}")
        return None

# Setup Chrome nhanh nhất
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--disable-extensions")
options.add_argument("--disable-images")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

driver = None
try:
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    print("✅ Chrome driver OK")
except Exception as e:
    print(f"❌ Lỗi driver: {e}")

if driver:
    search_queries = [
        "bãi đỗ xe Hà Nội", "parking Hà Nội", "bãi giữ xe máy Hà Nội",
        "bãi đỗ xe TP.HCM", "parking Sài Gòn", "bãi giữ xe máy TP.HCM",
        "bãi đỗ xe Đà Nẵng", "parking Đà Nẵng",
        "bãi đỗ xe Hải Phòng", "bãi đỗ xe Cần Thơ", "bãi đỗ xe Nha Trang",
        "bãi đỗ xe Vũng Tàu", "bãi đỗ xe Bình Dương", "bãi đỗ xe Đồng Nai",
        "bãi đỗ xe sân bay Nội Bài", "bãi đỗ xe sân bay Tân Sơn Nhất",
        "bãi đỗ xe trung tâm thương mại", "bãi đỗ xe Vincom", "bãi đỗ xe Big C", "bãi đỗ xe Aeon Mall",
        "bãi đỗ xe bệnh viện Bạch Mai", "bãi đỗ xe bệnh viện Chợ Rẫy",
        "bãi đỗ xe công cộng", "bãi đỗ xe ngầm",
    ]
    print(f"🎯 SẼ CHẠY {len(search_queries)} QUERIES - MỖI QUERY LƯU RIÊNG FILE")

    all_results = []
    total_saved_files = []
    # --- BỘ ĐẾM TOÀN CỤC ĐỂ TRÁNH TRÙNG LẶP GIỮA CÁC QUERY ---
    global_seen_urls = set()

    for query_idx, search_query in enumerate(search_queries, 1):
        print(f"\n{'='*100}")
        print(f"🚀 QUERY {query_idx}/{len(search_queries)}: '{search_query}'")
        print(f"{'='*100}")

        query_results = []
        encoded_query = urllib.parse.quote(search_query)
        url = f"https://www.google.com/maps/search/{encoded_query}/"

        try:
            print(f"🌐 Loading: {url}")
            driver.get(url)
            # Đợi cho đến khi phần tử chính của kết quả tìm kiếm xuất hiện
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//div[@role="feed"]'))
            )
            time.sleep(2)
        except Exception as e:
            print(f"❌ Lỗi load trang: {e}")
            continue

        # Cuộn để load NHIỀU kết quả
        try:
            scrollable_div = driver.find_element(By.XPATH, '//div[@role="feed"]')
            print("📜 Cuộn để load danh sách địa điểm...")

            # Cuộn 30 lần để lấy nhiều kết quả
            for i in range(30):
                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
                time.sleep(0.4)
                if (i + 1) % 10 == 0:
                    print(f"  📜 Đã cuộn {i+1}/30 lần...")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Cuộn lỗi: {e}")

        # --- CHIẾN LƯỢC MỚI: LẤY TẤT CẢ LINK TRƯỚC ---
        try:
            places = driver.find_elements(By.XPATH, '//a[contains(@href, "/maps/place/")]')
            place_links = [p.get_attribute('href') for p in places if p.get_attribute('href')]
            # Lọc các link trùng lặp trong query này
            unique_links = list(dict.fromkeys(place_links))
            print(f"🎯 Tìm thấy {len(unique_links)} địa điểmユニーク.")
        except Exception as e:
            print(f"❌ Không tìm thấy link địa điểm: {e}")
            continue

        # --- XỬ LÝ TỪNG LINK ĐÃ THU THẬP ---
        for place_idx, link in enumerate(unique_links):
            # Bỏ qua nếu link này đã được xử lý ở một query khác trước đó
            if link in global_seen_urls:
                print(f"[{place_idx+1}/{len(unique_links)}] ⏭️ Bỏ qua (đã xử lý): {link[:60]}...")
                continue

            print(f"🔍 [{place_idx+1}/{len(unique_links)}] Đang xử lý: {link[:80]}...")

            try:
                # Truy cập trực tiếp vào trang của địa điểm
                driver.get(link)
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.XPATH, '//h1')) # Đợi tên (h1) xuất hiện
                )
                time.sleep(1)

                # Lấy thông tin chi tiết
                name = driver.find_element(By.XPATH, '//h1').text.strip()
                lat, lng = extract_lat_lng_from_url(driver.current_url)

                address = "Không có địa chỉ"
                try:
                    # Selector ổn định hơn cho địa chỉ
                    address_elem = driver.find_element(By.CSS_SELECTOR, 'button[data-item-id="address"]')
                    address = address_elem.get_attribute('aria-label').strip()
                except:
                    pass

                rating = ""
                try:
                    # Selector cho rating
                    rating_elem = driver.find_element(By.CSS_SELECTOR, 'div.F7nice > span > span[aria-hidden="true"]')
                    rating_text = rating_elem.text.strip()
                    if rating_text: rating = rating_text
                except:
                    pass

                # Xác định thành phố
                city = "Không xác định"
                address_to_check = address.replace("Địa chỉ: ", "")
                for vietnam_city in VIETNAM_CITIES:
                    if vietnam_city.lower() in address_to_check.lower():
                        city = vietnam_city
                        break

                # Lấy reviews
                reviews = get_reviews(driver, max_reviews=20)

                result = {
                    "stt": len(query_results) + 1, "name": name, "address": address_to_check,
                    "lat": lat if lat else "", "lng": lng if lng else "",
                    "reviews": reviews, "rating": rating, "query": search_query, "city": city
                }
                query_results.append(result)
                global_seen_urls.add(link) # Đánh dấu link này đã được xử lý trên toàn cục
                print(f"    ✅ Thêm thành công: '{name}' - Rating: {rating} - City: {city}")

            except Exception as e:
                print(f"    ❌ Lỗi khi xử lý link {link}: {e}")
                continue

        # Kết thúc query này
        print(f"\n🏁 Query '{search_query}' HOÀN THÀNH. Xử lý thành công: {len(query_results)} địa điểm.")
        if query_results:
            saved_file = save_results_immediately(query_results, search_query)
            if saved_file:
                total_saved_files.append(saved_file)
            all_results.extend(query_results)

        print(f"📊 TỔNG CỘNG ĐẾN GIỜ: {len(all_results)} địa điểm.")
        if query_idx < len(search_queries):
            print("😴 Nghỉ 5 giây trước query tiếp theo...")
            time.sleep(5)

    # KẾT THÚC TẤT CẢ
    print(f"\n{'='*120}")
    print(f"🎉🎉 HOÀN TẤT TẤT CẢ {len(search_queries)} QUERIES! 🎉🎉")
    print(f"📊 Tổng cộng thu thập: {len(all_results)} địa điểmユニーク.")
    print(f"📁 Đã lưu: {len(total_saved_files)} files riêng biệt.")
    print(f"{'='*120}")

    # Lưu file tổng hợp cuối cùng
    if all_results:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_file = f"{DRIVE_PATH}/VIETNAM_PARKING_FINAL_ALL_{timestamp}.csv"

            # Lọc lại kết quả cuối cùng để đảm bảo không có trùng lặp tuyệt đối (dựa trên tên và địa chỉ)
            final_unique_results = []
            seen_final = set()
            for r in all_results:
                identifier = (r['name'], r['address'])
                if identifier not in seen_final:
                    final_unique_results.append(r)
                    seen_final.add(identifier)

            print(f"Filtered down to {len(final_unique_results)} truly unique results for final file.")

            with open(final_file, mode="w", encoding="utf-8", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["stt", "name", "address", "lat", "lng", "reviews", "rating", "query", "city"])
                writer.writeheader()
                for i, row in enumerate(final_unique_results, 1):
                    row_copy = row.copy()
                    row_copy["stt"] = i
                    row_copy["reviews"] = json.dumps(row["reviews"], ensure_ascii=False)
                    writer.writerow(row_copy)

            print(f"📄 FILE TỔNG HỢP CUỐI CÙNG: {final_file}")

        except Exception as e:
            print(f"❌ Lỗi lưu file tổng hợp: {e}")

    try:
        driver.quit()
        print("✅ Đã đóng browser")
    except:
        pass

print("\n🚀 SCRIPT HOÀN TẤT!")