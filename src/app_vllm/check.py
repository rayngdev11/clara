import json
from collections import defaultdict

def analyze_results(json_path, keywords):
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    keyword_counts = defaultdict(int)

    for setting, outputs in results.items():
        count = 0
        for text in outputs:
            if all(keyword.lower() in text.lower() for keyword in keywords):
                count += 1
        if count > 0:
            keyword_counts[setting] = count

    # Sắp xếp theo số lần match giảm dần
    sorted_results = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"🔍 Thống kê các tổ hợp sinh ra cả 2 cụm từ:")
    for setting, count in sorted_results:
        print(f"{setting} => ✅ {count}/{len(results[setting])} lần")

    return sorted_results
keywords = ["Van tim nhân tạo", "Chỉ thép lồng ngực"]
# keywords=["gãy cũ xương đòn trái"]
analyze_results(
    json_path="/home/datnvt/project/Medical_CLARA/infer/demo_clara/clara_setting_false_test4_multi.json",
    keywords=keywords
)



"""
test lần 1 ảnh test4.png ảnh van tim nhân tạo: thông số  ảnh 448 * 448

temp=0.9_top_p=1.0_top_k=0 => ✅ 4/5 lần
temp=0.7_top_p=1.0_top_k=30 => ✅ 3/5 lần
temp=0.8_top_p=0.8_top_k=70 => ✅ 3/5 lần
temp=0.8_top_p=0.9_top_k=75 => ✅ 3/5 lần
temp=0.8_top_p=0.9_top_k=85 => ✅ 3/5 lần
temp=0.9_top_p=0.6_top_k=75 => ✅ 3/5 lần
temp=0.9_top_p=0.8_top_k=15 => ✅ 3/5 lần
temp=0.9_top_p=0.8_top_k=40 => ✅ 3/5 lần
temp=0.9_top_p=0.9_top_k=15 => ✅ 3/5 lần
temp=0.9_top_p=1.0_top_k=40 => ✅ 3/5 lần
"""