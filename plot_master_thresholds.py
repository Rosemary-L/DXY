import pandas as pd
import spacy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_CSV = "/user/quzhengyi/dxy/1956-2025.csv"
OUTPUT_DIR = "/user/quzhengyi/dxy/output"
# 两张图分别输出为两个 PNG 文件（原四宫格中的 b、c）
OUTPUT_PLOT_LIFESPAN = os.path.join(OUTPUT_DIR, "Master_Thresholds_EDA_Lifespan_b.png")
OUTPUT_PLOT_POST1992 = os.path.join(OUTPUT_DIR, "Master_Thresholds_EDA_Post1992_c.png")

# 与 Image2.py 新兴议题「后冷战纯度」筛选阈值一致
POST_1992_RATIO_THRESHOLD = 0.95

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    "今年", "皆様", "人々", "地域", "関係", " মিলিটারি", "問題", "対応", "実現", "発展", "協力", 
    "推進", "取組", "取り組み", "改革", "政策", "分野", "状況", "必要", "日本国", 
    "全体", "国際社会", "時代", "会議", "平和", "皆様方"
}

# ==========================================
# 3. 加载模型与处理数据
# ==========================================
print("正在配置 GPU 并加载 Transformer 模型...")
spacy.require_gpu()
nlp = spacy.load("ja_core_news_trf")
nlp.max_length = 5000000

df_raw = pd.read_csv(INPUT_CSV, header=None).dropna(subset=[1])
entity_stats = {}

print("🚀 正在全局扫描文献以提取多维特征...")
for index, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
    year = int(str(row[0]).strip())
    text = str(row[1])

    if len(text) < 10:
        continue
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
                    entity_stats[entity_text] = {
                        "total_freq": 0,
                        "years_set": set(),
                        "post1992_freq": 0,
                    }

                stats = entity_stats[entity_text]
                stats["total_freq"] += 1
                stats["years_set"].add(year)
                if year >= 1992:
                    stats["post1992_freq"] += 1

# ==========================================
# 4. 计算 (b)(c) 两图所需指标
# ==========================================
print("\n🧮 正在计算生命周期与后冷战纯度指标...")
all_years_count = []
post1992_ratios = []

for ent, stats in entity_stats.items():
    total_freq = stats["total_freq"]
    years_count = len(stats["years_set"])

    all_years_count.append(years_count)

    if years_count >= 4:
        post1992_ratios.append(stats["post1992_freq"] / total_freq)

# ==========================================
# 5. 分别绘制并保存 (b)(c) 为两个 PNG
# ==========================================
sns.set_theme(style="whitegrid")
figsize_single = (9, 6)

# ---------- (b) 实体生命周期分布 ----------
print("🎨 正在绘制并保存 (b) 生命周期图...")
fig_b, ax_b = plt.subplots(figsize=figsize_single)
sns.histplot(all_years_count, bins=range(1, 71, 2), color="#55A868", edgecolor="black", ax=ax_b, alpha=0.7)
ax_b.axvline(x=4, color="#8172B3", linestyle=":", linewidth=2.5, label="Emerging Min (4 Yrs)")
ax_b.axvline(x=5, color="#C44E52", linestyle="--", linewidth=2.5, label="Phase Min (5 Yrs)")
ax_b.axvline(x=35, color="#DD8452", linestyle="-.", linewidth=2.5, label="Core Min (35 Yrs)")
ax_b.set_title("Lifespan Distribution (Years Appeared)", fontsize=16, fontweight="bold")
ax_b.set_xlabel("Number of Years Appeared (1-70)", fontsize=14)
ax_b.set_ylabel("Number of Entities", fontsize=14)
ax_b.legend(fontsize=12)
plt.tight_layout()
fig_b.savefig(OUTPUT_PLOT_LIFESPAN, dpi=300, bbox_inches="tight")
plt.close(fig_b)

# ---------- (c) 后冷战纯度分布 ----------
print("🎨 正在绘制并保存 (c) 后冷战纯度图...")
fig_c, ax_c = plt.subplots(figsize=figsize_single)
sns.histplot(post1992_ratios, bins=20, kde=True, color="#8172B3", edgecolor="black", ax=ax_c, alpha=0.7)
ax_c.axvline(
    x=POST_1992_RATIO_THRESHOLD,
    color="#C44E52",
    linestyle="--",
    linewidth=2.5,
    label=f"Threshold ({int(POST_1992_RATIO_THRESHOLD * 100)}%)",
)
ax_c.set_title("Post-1992 Frequency Ratio (Bimodal Valley)", fontsize=16, fontweight="bold")
ax_c.set_xlabel("Ratio of Post-1992 Frequency to Total Frequency", fontsize=14)
ax_c.set_ylabel("Number of Entities", fontsize=14)
ax_c.set_xticks(np.arange(0, 1.1, 0.1))
ax_c.set_xticklabels([f"{int(x * 100)}%" for x in np.arange(0, 1.1, 0.1)])
ax_c.legend(fontsize=12, loc="upper left")
plt.tight_layout()
fig_c.savefig(OUTPUT_PLOT_POST1992, dpi=300, bbox_inches="tight")
plt.close(fig_c)

print("🎉 两张图已分别保存：")
print(f"   (b) -> {OUTPUT_PLOT_LIFESPAN}")
print(f"   (c) -> {OUTPUT_PLOT_POST1992}")