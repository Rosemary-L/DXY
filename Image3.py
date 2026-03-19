import pandas as pd
import spacy
from tqdm import tqdm
import os

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_CSV = "/user/quzhengyi/dxy/1956-2025.csv"
OUTPUT_EXCEL = "/user/quzhengyi/dxy/output/Image3_Phase_Entities.xlsx"

os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)

# ==========================================
# 2. 核心配置：归一化字典与停用词清单
# ==========================================
TARGET_LABELS = {"GPE", "ORG", "LAW", "EVENT", "NORP", "LOC", "MOVEMENT", "PRODUCT"}

ENTITY_ALIASES = {
    "国際連合": "国連", "安全保障理事会": "安保理", "国連安保理": "安保理",
    "国際連合憲章": "国連憲章", "国連憲章再検討問題": "国連憲章", "国連憲章調印": "国連憲章", "国連憲章の原則": "国連憲章",
    "ＰＫＯ": "PKO", "国連ＰＫＯ": "PKO", "国連平和維持活動": "PKO",
    "核兵器不拡散条約": "核不拡散条約(NPT)", "核不拡散条約": "核不拡散条約(NPT)",
    "核兵器拡散防止条約": "核不拡散条約(NPT)", "核拡散防止条約": "核不拡散条約(NPT)", "ＮＰＴ": "核不拡散条約(NPT)",
    "核兵器不拡散条約体制": "核不拡散体制", "核不拡散条約体制": "核不拡散体制", "核不拡散": "核不拡散体制",
    "核軍縮 ・ 不拡散体制": "核不拡散体制", "ＮＰＴ体制": "核不拡散体制",
    "核兵器実験禁止条約": "核実験禁止条約", "全面核実験禁止条約": "核実験禁止条約",
    "全面核実験禁止条約交渉": "核実験禁止条約", "核実験停止条約": "核実験禁止条約",
    "核兵器実験": "核実験", "核実験全面禁止": "核実験", "核実験検証制度": "核実験",
    "核実験２条約": "核実験", "部分核停条約": "核実験",
    "核実験中止協定": "核実験中止", "核実験停止協定": "核実験中止", "核兵器実験停止協定": "核実験中止",
    "ジュネーヴ核実験中止会議": "核実験中止", "ジュネーブ核実験中止会議": "核実験中止",
    "核実験中止問題": "核実験中止", "核実験中止会議": "核実験中止",
    "核兵器戦争": "核戦争", "中距離核戦力交渉": "核戦争",
    "中近東": "中東", "中東地域": "中東", "アフガン": "アフガニスタン",
    "カンボジア": "カンボディア", "ヴィエトナム": "ベトナム", "ヴィエトナム社会主義共和国": "ベトナム",
    "太平洋地域": "太平洋", "南アフリカ共和国": "南アフリカ",
    "日本国政府": "日本政府", "日本国代表団": "日本代表団",
    "ＴＩＣＡＤ": "TICAD", "TICADIV": "TICAD", "ＯＤＡ": "ODA",
    "ＵＨＣ": "UHC", "ユニバーサル ・ ヘルス ・ カバレッジ": "UHC",
    "ＭＤＧｓ": "MDGs", "ＳＤＧｓ": "SDGs", "UNウィメン": "UN Women",
    "米国": "アメリカ", "コロナ": "新型コロナ", "新型コロナウイルス": "新型コロナ",
    "中共": "中国", "ソ連": "ソビエト連邦", "露": "ロシア"
}

POLITICAL_STOP_WORDS = {
    "日本", "我が国", "世界", "国民", "政府", "国家", "国", 
    "社会", "経済", "今日", "今年", "皆様", "人々", "地域", 
    "関係", "問題", "対応", "実現", "発展", "協力", "推進", 
    "取組", "取り組み", "改革", "政策", "分野", "状況", "必要", 
    "日本国", "全体", "国際社会", "時代", "会議", "平和", "皆様方"
}

# ==========================================
# 3. 加载 GPU 与模型
# ==========================================
print("正在配置 GPU 并加载 Transformer 模型...")
spacy.require_gpu()
nlp = spacy.load("ja_core_news_trf")
nlp.max_length = 5000000

# ==========================================
# 4. 执行全局提取，保留年份细节
# ==========================================
print("📥 正在读取原始讲稿数据...")
df_raw = pd.read_csv(INPUT_CSV, header=None).dropna(subset=[1])

# 结构: { 实体: { 年份: 频次 } }
entity_yearly_freq = {}
years_in_doc = set()

print("🚀 开始扫描历史文献，捕获时间轨迹...")
for index, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
    year = int(str(row[0]).strip())
    text = str(row[1])
    years_in_doc.add(year)
    
    if len(text) < 10: continue
        
    doc = nlp(text)
    
    for ent in doc.ents:
        label = ent.label_
        if ent.text.strip() == "アフリカ" and label == "GPE":
            label = "LOC"
            
        if label in TARGET_LABELS:
            entity_text = ent.text.strip()
            if entity_text in ENTITY_ALIASES:
                entity_text = ENTITY_ALIASES[entity_text]
                
            if len(entity_text) > 1 and entity_text not in POLITICAL_STOP_WORDS:
                if entity_text not in entity_yearly_freq:
                    entity_yearly_freq[entity_text] = {"Label": label, "Yearly_Data": {}}
                
                # 累加该年频次
                current_count = entity_yearly_freq[entity_text]["Yearly_Data"].get(year, 0)
                entity_yearly_freq[entity_text]["Yearly_Data"][year] = current_count + 1

# ==========================================
# 5. 执行【阶段性形象】三重逻辑筛选
# ==========================================
print("\n🧮 正在执行【阶段性形象】筛选 (5年 <= 出场 <= 40%，且 10年聚集度 >= 70%)...")

total_unique_years = len(years_in_doc)
max_years_threshold = int(total_unique_years * 0.40) # 40% 的上限
print(f"📌 数据集总年份: {total_unique_years}年，40% 时代局限性上限: {max_years_threshold}年")

phase_results = []

for ent, data in entity_yearly_freq.items():
    yearly_data = data["Yearly_Data"]
    label = data["Label"]
    
    total_freq = sum(yearly_data.values())
    years_list = sorted(yearly_data.keys())
    years_count = len(years_list)
    
    # 【指标 1】：出场年份数 >= 5 年
    if years_count < 5:
        continue
        
    # 【指标 2】：出场年份数 <= 总年份的 40%
    if years_count > max_years_threshold:
        continue
        
    # 【指标 3】：计算 10年高光期聚集度
    max_10y_freq = 0
    best_10y_start = 0
    
    # 滑动窗口扫描 (假设范围是 1956 到 2025)
    min_yr, max_yr = min(years_in_doc), max(years_in_doc)
    for start_year in range(min_yr, max_yr - 8): 
        end_year = start_year + 9
        # 统计这 10 年内的总频次
        window_freq = sum(yearly_data.get(y, 0) for y in range(start_year, end_year + 1))
        
        if window_freq > max_10y_freq:
            max_10y_freq = window_freq
            best_10y_start = start_year
            
    # 计算最大聚集度
    concentration_ratio = max_10y_freq / total_freq
    
    # 聚集度 >= 70% 视为“高度集中”
    if concentration_ratio >= 0.70:
        phase_results.append({
            "Entity": ent,
            "Label": label,
            "Total_Frequency": total_freq,
            "Years_Count": years_count,
            "Peak_Decade": f"{best_10y_start}-{best_10y_start+9}",
            "Concentration_Ratio": f"{concentration_ratio:.1%}",
            "Appeared_Years_List": ", ".join(map(str, years_list))
        })

# 转化为 DataFrame，按“总频次”降序
df_phase = pd.DataFrame(phase_results)

if not df_phase.empty:
    df_phase = df_phase.sort_values(by="Total_Frequency", ascending=False)
    df_phase.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n🎉 筛选完成！共有 {len(df_phase)} 个实体成功入选【阶段性形象】！")
    print(f"✅ 结果已保存至: {OUTPUT_EXCEL}")
else:
    print("\n⚠️ 哎呀，没有筛选出符合条件的词，可能需要放宽聚集度（比如降到 60%）。")