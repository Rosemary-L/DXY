import pandas as pd
import os

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_CSV = "/user/quzhengyi/dxy/output/overall_top15.csv"

# 我们生成两个 Excel，一个是原样转换，一个是排版好的矩阵
OUTPUT_EXCEL_RAW = "/user/quzhengyi/dxy/output/overall_top15.xlsx"
OUTPUT_EXCEL_MATRIX = "/user/quzhengyi/dxy/output/Top15_Matrix.xlsx"

print("📥 正在读取刚才提取的 CSV 数据...")
df = pd.read_csv(INPUT_CSV)

# ==========================================
# 2. 基础转换：直接把原始的长表保存为 Excel
# ==========================================
df.to_excel(OUTPUT_EXCEL_RAW, index=False)
print(f"✅ [基础版] 原始长表已成功转为 Excel: \n   -> {OUTPUT_EXCEL_RAW}")

# ==========================================
# 3. 进阶排版：生成“年份 x 1~15名”的二维矩阵表
# ==========================================
print("\n🪄 正在施展魔法，构建二维矩阵排版...")
# 拼接单元格内容，例如：新型コロナ (EVENT, 58)
df['Cell_Content'] = df['Entity'] + " (" + df['Label'] + ", " + df['Frequency'].astype(str) + ")"

# 按照年份分组，自动打上 1 到 15 的排名
df['Rank'] = df.groupby('Year').cumcount() + 1

# 数据透视：把 Year 变成行，Rank 变成列
matrix_df = df.pivot(index='Year', columns='Rank', values='Cell_Content')

# 确保有 1 到 15 列
matrix_df = matrix_df.reindex(columns=range(1, 16))

# 保存矩阵版 Excel
matrix_df.to_excel(OUTPUT_EXCEL_MATRIX)
print(f"✅ [终极版] 二维排版矩阵 Excel 已生成！(强烈推荐看这个): \n   -> {OUTPUT_EXCEL_MATRIX}")