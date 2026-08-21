import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap
import plotly.express as px

def main():
    print("[*] Veriler ve embedding dosyası yükleniyor...")
    
    emb_path = "embeddings/mpnet_multilingual_embeddings.npy"
    csv_path = "data/balanced_articles.csv"
    anom_path = "results/hdbscan_anomaliler.csv"

    if not os.path.exists(emb_path) or not os.path.exists(anom_path):
        print(f"[!] Hata: Dosyalar bulunamadı.")
        return

    embeddings = np.load(emb_path)
    df = pd.read_csv(csv_path)
    df_anom = pd.read_csv(anom_path)

    print(f"[*] Toplam {len(embeddings)} makale 2D UMAP uzayına izdüşürülüyor...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    coords_2d = reducer.fit_transform(embeddings)

    df['x'] = coords_2d[:, 0]
    df['y'] = coords_2d[:, 1]
    
    anom_dict = df_anom.set_index('external_id').to_dict(orient='index')
    
    status_list = []
    hover_texts = []
    
    for _, row in df.iterrows():
        ext_id = row.get('external_id')
        if ext_id in anom_dict:
            a_info = anom_dict[ext_id]
            oncelik = a_info.get('oncelik', 'Anomali')
            status_list.append(oncelik)
            hover_texts.append(
                f"<b>ID:</b> {ext_id}<br>"
                f"<b>Başlık:</b> {str(row.get('baslik', ''))[:80]}...<br>"
                f"<b>Mevcut:</b> {a_info.get('mevcut_kategori', '')}<br>"
                f"<b>Öneri:</b> {a_info.get('oneri_kategori', '')}<br>"
                f"<b>Risk Skoru:</b> {a_info.get('risk_skoru', 0):.3f}"
            )
        else:
            status_list.append("Normal Makale")
            hover_texts.append(f"ID: {ext_id}<br>{str(row.get('baslik', ''))[:60]}")

    df['Durum'] = status_list
    df['Detay'] = hover_texts

    os.makedirs("results", exist_ok=True)

    # 1. İnteraktif Plotly HTML Grafiği
    print("[*] İnteraktif HTML grafiği oluşturuluyor...")
    color_map = {
        "Normal Makale": "#bdc3c7",
        "YÜKSEK (Farklı Alt Alan)": "#f39c12",
        "KRİTİK (Farklı Ana Disiplin)": "#e74c3c"
    }

    fig = px.scatter(
        df, x='x', y='y', color='Durum',
        color_discrete_map=color_map,
        custom_data=['Detay'],
        title='TR Dizin 2D UMAP Projeksiyonu ve Anomali Dağılımı',
        opacity=0.75,
        width=1200, height=800
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        marker=dict(size=5)
    )
    fig.write_html("results/umap_interactive_plot.html")
    print("[+] İnteraktif grafik kaydedildi: results/umap_interactive_plot.html")

    # 2. Statik PNG Çıktısı
    print("[*] Statik PNG grafiği çiziliyor...")
    plt.figure(figsize=(14, 10), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')

    normal_mask = df['Durum'] == 'Normal Makale'
    plt.scatter(df[normal_mask]['x'], df[normal_mask]['y'], c='#bdc3c7', s=10, alpha=0.3, label='Normal Makaleler')

    yuksek_mask = df['Durum'] == 'YÜKSEK (Farklı Alt Alan)'
    plt.scatter(df[yuksek_mask]['x'], df[yuksek_mask]['y'], c='#f39c12', s=45, alpha=0.85, edgecolors='black', linewidth=0.5, label=f"Yüksek Öncelik ({yuksek_mask.sum()} adet)")

    kritik_mask = df['Durum'] == 'KRİTİK (Farklı Ana Disiplin)'
    plt.scatter(df[kritik_mask]['x'], df[kritik_mask]['y'], c='#e74c3c', s=70, alpha=0.95, edgecolors='black', linewidth=0.8, marker='X', label=f"Kritik Öncelik ({kritik_mask.sum()} adet)")

    plt.title('TR Dizin 2D UMAP Kümeleme & HDBSCAN Anomali Uzayı', fontsize=14, fontweight='bold')
    plt.xlabel('UMAP 1', fontsize=11)
    plt.ylabel('UMAP 2', fontsize=11)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig("results/umap_static_scatter.png", dpi=300)
    plt.close()
    print("[+] Statik grafik kaydedildi: results/umap_static_scatter.png")

if __name__ == "__main__":
    main()