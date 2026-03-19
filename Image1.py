import pandas as pd
import spacy
from tqdm import tqdm
import os

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_CSV = "/user/quzhengyi/dxy/1956-2025.csv"
OUTPUT_EXCEL = "/user/quzhengyi/dxy/output/Image1_Core_Entities2.xlsx"

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

# 政治废话排除清单
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
# 4. 执行全局提取与清洗
# ==========================================
print("📥 正在读取原始讲稿数据...")
df_raw = pd.read_csv(INPUT_CSV, header=None).dropna(subset=[1])

global_stats = {}

print("🚀 开始扫描历史文献，构建实体网络...")
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
                
            # 过滤单字与政治废话
            if len(entity_text) > 1 and entity_text not in POLITICAL_STOP_WORDS:
                if entity_text not in global_stats:
                    global_stats[entity_text] = {"Label": label, "Total_Freq": 0, "Years_Appeared": set()}
                
                global_stats[entity_text]["Total_Freq"] += 1
                global_stats[entity_text]["Years_Appeared"].add(year)

# ==========================================
# 5. 核心形象筛选：出现年数 ≥ 35（仅此一条）
# ==========================================
CORE_MIN_YEARS = 35
print(f"\n🧮 正在执行【核心形象】筛选：出现年数 ≥ {CORE_MIN_YEARS} 年...")

rows = []
for ent, stats in global_stats.items():
    years_sorted = sorted(stats["Years_Appeared"])
    rows.append({
        "Entity": ent,
        "Label": stats["Label"],
        "Total_Frequency": stats["Total_Freq"],
        "Years_Count": len(stats["Years_Appeared"]),
        "Appeared_Years_List": ", ".join(map(str, years_sorted)),
    })

df_all = pd.DataFrame(rows)
df_core_image = df_all[df_all["Years_Count"] >= CORE_MIN_YEARS].copy()
# 行序：按总频次降序（简单默认）
df_core_image = df_core_image.sort_values(by="Total_Frequency", ascending=False)

out_cols = ["Entity", "Label", "Total_Frequency", "Years_Count", "Appeared_Years_List"]
df_core_image[out_cols].to_excel(OUTPUT_EXCEL, index=False)

print(f"\n🎉 筛选完成！共有 {len(df_core_image)} 个实体入选【核心形象】名单！")
print(f"✅ 结果已保存至: {OUTPUT_EXCEL}")