import pandas as pd
import spacy
from tqdm import tqdm
import os

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_CSV = "/user/quzhengyi/dxy/1956-2025.csv"
OUTPUT_EXCEL = "/user/quzhengyi/dxy/output/Image2_Emerging_Entities.xlsx"

# 新兴议题形象：1992 年后频次占比阈值（与 plot_master_thresholds 图中红线一致）
POST_1992_RATIO_THRESHOLD = 0.95

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
    "ソ連": "ソビエト連邦", "露": "ロシア"
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
# 4. 执行全局提取，统计年份分布
# ==========================================
print("📥 正在读取原始讲稿数据...")
df_raw = pd.read_csv(INPUT_CSV, header=None).dropna(subset=[1])

# 数据结构: { 实体: {"Total_Freq": 0, "Years_Set": set(), "Freq_Post_1992": 0} }
entity_stats = {}

print("🚀 开始扫描历史文献，追踪冷战后轨迹...")
for index, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
    year = int(str(row[0]).strip())
    text = str(row[1])
    
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
                if entity_text not in entity_stats:
                    entity_stats[entity_text] = {"Label": label, "Total_Freq": 0, "Years_Set": set(), "Freq_Post_1992": 0}
                
                # 记录总频次和出场年份
                entity_stats[entity_text]["Total_Freq"] += 1
                entity_stats[entity_text]["Years_Set"].add(year)
                
                # 记录 1992 年及以后的频次
                if year >= 1992:
                    entity_stats[entity_text]["Freq_Post_1992"] += 1

# ==========================================
# 5. 执行【新兴议题】极简双指标筛选
# ==========================================
print(f"\n🧮 正在执行【后冷战新贵】指标筛选 (≥4年，且 ≥{POST_1992_RATIO_THRESHOLD:.0%}能量在1992年后)...")

emerging_results = []
for ent, stats in entity_stats.items():
    total_freq = stats["Total_Freq"]
    years_count = len(stats["Years_Set"])
    freq_post_1992 = stats["Freq_Post_1992"]
        
    # 指标 1：出场年份数 >= 4
    if years_count < 4:
        continue
        
    # 指标 2：1992年后能量占比 >= POST_1992_RATIO_THRESHOLD
    ratio_post_1992 = freq_post_1992 / total_freq
    if ratio_post_1992 >= POST_1992_RATIO_THRESHOLD:
        first_year_appeared = min(stats["Years_Set"])
        
        emerging_results.append({
            "Entity": ent,
            "Label": stats["Label"],
            "Total_Frequency": total_freq,
            "Years_Count": years_count,
            "First_Year": first_year_appeared,
            "Post_1992_Ratio": f"{ratio_post_1992:.1%}", 
            "Appeared_Years_List": ", ".join(map(str, sorted(list(stats["Years_Set"]))))
        })

# 转化为 DataFrame，按总频次降序
df_emerging = pd.DataFrame(emerging_results)

if not df_emerging.empty:
    df_emerging = df_emerging.sort_values(by="Total_Frequency", ascending=False)
    df_emerging.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n🎉 筛选完成！共有 {len(df_emerging)} 个实体成功入选【新兴议题形象】！")
    print(f"✅ 结果已保存至: {OUTPUT_EXCEL}")
else:
    print("\n⚠️ 哎呀，按这个标准没有筛选出任何词。你可以试着调整一下年份下限或比例。")