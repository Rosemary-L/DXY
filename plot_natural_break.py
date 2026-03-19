import pandas as pd
import spacy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_CSV = "/user/quzhengyi/dxy/1956-2025.csv"
OUTPUT_PLOT = "/user/quzhengyi/dxy/output/Image3_Natural_Break_EDA.png"
# 🚨 新增：将分布图背后的具体数据导出为 Excel
OUTPUT_EXCEL_DATA = "/user/quzhengyi/dxy/output/Image3_Natural_Break_Data.xlsx"

os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)

# ==========================================
# 2. 核心配置：归一化字典与停用词清单 
# ==========================================
TARGET_LABELS = {"GPE", "ORG", "LAW", "EVENT", "NORP", "LOC", "MOVEMENT", "PRODUCT"}

ENTITY_ALIASES = {
    "国際連合": "国連", "安全保障理事会": "安保理", "国連安保理": "安保理",
    "国際連合憲章": "国連憲章", "ＰＫＯ": "PKO", "国連ＰＫＯ": "PKO", "国連平和維持活動": "PKO",
    "核兵器不拡散条約": "核不拡散条約(NPT)", "核不拡散条約": "核不拡散条約(NPT)", "ＮＰＴ": "核不拡散条約(NPT)",
    "核兵器実験禁止条約": "核実験禁止条約", "全面核実験禁止条約": "核実験禁止条約",
    "核兵器実験": "核実験", "核兵器戦争": "核戦争",
    "中近東": "中東", "アフガン": "アフガニスタン", "カンボジア": "カンボディア", 
    "ヴィエトナム": "ベトナム", "太平洋地域": "太平洋", "南アフリカ共和国": "南アフリカ",
    "日本国政府": "日本政府", "ＴＩＣＡＤ": "TICAD", "ＯＤＡ": "ODA", "ＵＨＣ": "UHC", 
    "ＭＤＧｓ": "MDGs", "ＳＤＧｓ": "SDGs", "米国": "アメリカ", "コロナ": "新型コロナ",
    "新型コロナウイルス": "新型コロナ", "中共": "中国", "ソ連": "ソビエト連邦", "露": "ロシア"
}

POLITICAL_STOP_WORDS = {
    "日本", "我が国", "世界", "国民", "政府", "国家", "国", "社会", "経済", "今日", 
    "今年", "皆様", "人々", "地域", "関係", "問題", "対応", "実現", "発展", "協力", 
    "推進", "取組", "取り組み", "改革", "政策", "分野", "状況", "必要", "日本国", 
    "全体", "国際社会", "時代", "会議", "平和", "皆様方"
}

# ==========================================
# 3. 加载模型与数据
# ==========================================
print("正在配置 GPU 并加载 Transformer 模型...")
spacy.require_gpu()
nlp = spacy.load("ja_core_news_trf")
nlp.max_length = 5000000

df_raw = pd.read_csv(INPUT_CSV, header=None).dropna(subset=[1])
entity_yearly_freq = {}
years_in_doc = set()

print("🚀 正在扫描文献以获取数据分布...")
for index, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
    year = int(str(row[0]).strip())
    text = str(row[1])
    years_in_doc.add(year)
    
    if len(text) < 10: continue
    doc = nlp(text)
    
    for ent in doc.ents:
        label = ent.label_
        if ent.text.strip() == "アフリカ" and label == "GPE": label = "LOC"
        if label in TARGET_LABELS:
            entity_text = ent.text.strip()
            if entity_text in ENTITY_ALIASES: entity_text = ENTITY_ALIASES[entity_text]
            if len(entity_text) > 1 and entity_text not in POLITICAL_STOP_WORDS:
                if entity_text not in entity_yearly_freq:
                    entity_yearly_freq[entity_text] = {"Yearly_Data": {}}
                entity_yearly_freq[entity_text]["Yearly_Data"][year] = entity_yearly_freq[entity_text]["Yearly_Data"].get(year, 0) + 1

# ==========================================
# 4. 计算并提取所有实体的完整数据字典
# ==========================================
print("\n🧮 正在计算聚集度并生成数据溯源表...")
total_unique_years = len(years_in_doc)
max_years_threshold = int(total_unique_years * 0.40)

# 🚨 新增：用一个列表保存所有词的具体数据
entity_ratio_data = []

for ent, data in entity_yearly_freq.items():
    yearly_data = data["Yearly_Data"]
    total_freq = sum(yearly_data.values())
    years_list = sorted(yearly_data.keys())
    years_count = len(years_list)
    
    if 5 <= years_count <= max_years_threshold:
        max_10y_freq = 0
        best_10y_start = 0
        min_yr, max_yr = min(years_in_doc), max(years_in_doc)
        
        for start_year in range(min_yr, max_yr - 8): 
            end_year = start_year + 9
            window_freq = sum(yearly_data.get(y, 0) for y in range(start_year, end_year + 1))
            if window_freq > max_10y_freq:
                max_10y_freq = window_freq
                best_10y_start = start_year
                
        ratio = max_10y_freq / total_freq
        
        # 将词汇、具体比例、最火十年等信息存起来
        entity_ratio_data.append({
            "Entity": ent,
            "Concentration_Ratio": ratio,
            "Concentration_Percentage": f"{ratio:.1%}", # 带百分号方便看
            "Peak_Decade": f"{best_10y_start}-{best_10y_start+9}",
            "Total_Frequency": total_freq,
            "Years_Count": years_count
        })

# 转化为 DataFrame 并保存为 Excel
df_data = pd.DataFrame(entity_ratio_data)
# 按聚集度从大到小排序！
df_data = df_data.sort_values(by="Concentration_Ratio", ascending=False)
df_data.to_excel(OUTPUT_EXCEL_DATA, index=False)
print(f"✅ 底层数据溯源表已生成: \n   -> {OUTPUT_EXCEL_DATA}")

# ==========================================
# 5. 使用 seaborn 绘制精美学术直方图
# ==========================================
print("\n🎨 正在绘制【自然断点分布图】...")

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

# 画图时直接使用 DataFrame 里的数值列
ax = sns.histplot(
    df_data["Concentration_Ratio"], 
    bins=25, 
    kde=True, 
    color="#4C72B0", 
    edgecolor="black", 
    alpha=0.7,
    line_kws={'linewidth': 2}
)

# 假设你最后决定把阈值定在 75% 
plt.axvline(x=0.75, color='#C44E52', linestyle='--', linewidth=3, label='Threshold (75%): Natural Break')

plt.text(0.77, ax.get_ylim()[1]*0.9, 'Phase Entities\n(Highly Concentrated)', color='#C44E52', fontsize=12, fontweight='bold')
plt.text(0.55, ax.get_ylim()[1]*0.9, 'General Entities\n(Dispersed)', color='#4C72B0', fontsize=12, fontweight='bold')

plt.title("Distribution of 10-Year Concentration Ratios for Political Entities", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Max 10-Year Concentration Ratio", fontsize=14, labelpad=10)
plt.ylabel("Frequency (Number of Entities)", fontsize=14, labelpad=10)

plt.xticks([i/10 for i in range(2, 11)], [f"{i*10}%" for i in range(2, 11)], fontsize=12)
plt.yticks(fontsize=12)

plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()

plt.savefig(OUTPUT_PLOT, dpi=300)
print(f"🎉 绘图大功告成！超高清学术图表已保存至: \n   -> {OUTPUT_PLOT}")