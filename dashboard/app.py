import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, Response, jsonify, render_template, request, send_file

from data_loader import load_algorithm_data, load_article_detail, BASE_DIR, RESULTS_DIR

app = Flask(__name__)


@app.route('/')
def home():
  return render_template('index.html')


@app.route('/api/anomalies', methods=['GET'])
def get_anomalies():
  algorithm = request.args.get('algorithm', 'hdbscan').lower()
  sort_order = request.args.get('sort', 'desc')
  priority_filter = request.args.get('priority', 'ALL')
  search_query = request.args.get('search', '').lower().strip()

  try:
    page = max(1, int(request.args.get('page', 1)))
  except (ValueError, TypeError):
    page = 1

  try:
    per_page = max(1, min(200, int(request.args.get('per_page', 50))))
  except (ValueError, TypeError):
    per_page = 50

  df = load_algorithm_data(algorithm)

  # --- EKLENECEK GÜVENLİK ÖNLEMLERİ ---
  if not df.empty:
    if 'risk_skoru' not in df.columns:
      df['risk_skoru'] = 0.5
    
    if 'oncelik' not in df.columns:
      df['oncelik'] = np.where(df['risk_skoru'] > 0.7, 'KRİTİK', 'NORMAL')

    if 'baslik' not in df.columns:
      df['baslik'] = df.get('title', 'Başlık Belirtilmemiş')

    if 'mevcut_kategori' not in df.columns:
      df['mevcut_kategori'] = df.get('keywords', 'Belirtilmemiş')

    if 'oneri_kategori' not in df.columns:
      df['oneri_kategori'] = 'Uyumlu / Normal'
      
    if 'external_id' not in df.columns:
      df['external_id'] = ''
  # ------------------------------------

  if df.empty:
    return jsonify({
        'data': [],
        'items': [],
        'stats': {
            'total_anomalies': 0,
            'avg_risk': 0,
            'critical_count': 0,
            'system_info': f'{algorithm.upper()} sonuç dosyası bulunamadı.',
        },
        'page': page,
        'per_page': per_page,
        'total': 0,
        'total_pages': 0,
    })

  # HDBSCAN Anomali Filtreleme (Yalnızca pipeline tarafından tespit edilen 388 anomali)
  if algorithm == 'hdbscan' and not df.empty:
    anom_file = os.path.join(RESULTS_DIR, 'hdbscan_anomaliler.csv')
    if os.path.exists(anom_file):
      anom_ids = pd.read_csv(anom_file, dtype={'external_id': str}, usecols=['external_id'])['external_id'].astype(str).str.strip()
      df = df[df['external_id'].astype(str).str.strip().isin(set(anom_ids))]
    elif 'supheli_mi' in df.columns:
      mask_ana = (df.get('ortak_agac_derinligi') == 0) & (df.get('knn_baskinlik', 0) >= 0.30)
      mask_alt = (df.get('ortak_agac_derinligi') == 1) & (
          (df.get('oneri_kategori') == df.get('knn_oneri'))
          | (df.get('knn_baskinlik', 0) >= 0.40)
      )
      df = df[
          (df['supheli_mi'] == 1)
          & (df.get('label_sim_fark', 0) >= 0.09)
          & (mask_ana | mask_alt)
          & (~df['baslik'].astype(str).str.strip().isin(['', '-', 'None', 'nan']))
          & (df['baslik'].astype(str).str.strip().str.len() > 3)
      ]

  # Filtrelemeler
  if priority_filter != 'ALL':
    df = df[df['oncelik'].astype(str).str.contains(priority_filter, case=False, na=False)]

  if search_query:
    df = df[
        df['baslik'].astype(str).str.lower().str.contains(search_query, na=False)
        | df['mevcut_kategori'].astype(str).str.lower().str.contains(search_query, na=False)
        | df['oneri_kategori'].astype(str).str.lower().str.contains(search_query, na=False)
        | df['external_id'].astype(str).str.lower().str.contains(search_query, na=False)
    ]

  # Sıralama
  ascending = sort_order == 'asc'
  if 'risk_skoru' in df.columns:
    df = df.sort_values(by='risk_skoru', ascending=ascending)

  total_count = len(df)
  total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 0

  stats = {
      'total_anomalies': total_count,
      'avg_risk': (
          round(float(df['risk_skoru'].mean()), 3)
          if not df.empty and 'risk_skoru' in df.columns
          else 0
      ),
      'critical_count': (
          int(df['oncelik'].astype(str).str.contains('KRİTİK', na=False).sum())
          if 'oncelik' in df.columns
          else 0
      ),
      'system_info': (
          'HDBSCAN Yoğunluk Tabanlı Anomali Modülü'
          if algorithm == 'hdbscan'
          else 'K-Means Centroid & Semantik Uyuşmazlık Modülü'
      ),
  }

  offset = (page - 1) * per_page
  paginated_df = df.iloc[offset:offset + per_page]
  paginated_records = paginated_df.to_dict(orient='records')

  return jsonify({
      'data': paginated_records,
      'items': paginated_records,
      'stats': stats,
      'page': page,
      'per_page': per_page,
      'total': total_count,
      'total_pages': total_pages,
  })

@app.route('/api/plot', methods=['GET'])
def get_plot():
  algorithm = request.args.get('algorithm', 'hdbscan').lower()
  df = load_algorithm_data(algorithm)

  # Risk skoru eşleme
  if 'risk_skoru' not in df.columns or df['risk_skoru'].fillna(0).sum() == 0:
    for c in ['risk', 'risk_score', 'anomaly_score', 'outlier_score', 'score']:
      if c in df.columns:
        df['risk_skoru'] = df[c]
        break

  if 'risk_skoru' not in df.columns:
    df['risk_skoru'] = 0.5

  # Küme eşleme
  if 'kume' not in df.columns:
    if 'hdbscan_kume' in df.columns:
      df['kume'] = df['hdbscan_kume']
    elif 'kmeans_kume' in df.columns:
      df['kume'] = df['kmeans_kume']
    else:
      df['kume'] = -1

  # External ID eşleme
  if 'external_id' not in df.columns:
    for c in ['id', 'ArticleID', 'makale_id']:
      if c in df.columns:
        df['external_id'] = df[c]
        break
    if 'external_id' not in df.columns:
      df['external_id'] = df.index.astype(str)

  if df.empty:
    fig = go.Figure()
    fig.update_layout(title='Görüntülenecek veri bulunamadı.')
  else:
    if (
        'umap_x' not in df.columns
        or 'umap_y' not in df.columns
        or df['umap_x'].isna().any()
    ):
      np.random.seed(42)
      df['umap_x'] = np.random.normal(loc=15.0, scale=8.0, size=len(df))
      df['umap_y'] = np.random.normal(loc=15.0, scale=8.0, size=len(df))

    # Yalnızca minimum gerekli alanları içeren hafif DataFrame
    plot_df = pd.DataFrame({
        'external_id': df['external_id'].astype(str),
        'umap_x': df['umap_x'].fillna(0.0),
        'umap_y': df['umap_y'].fillna(0.0),
        'risk_skoru': df['risk_skoru'].fillna(0.5),
        'kume': df['kume'].fillna(-1).astype(int),
    })

    records = plot_df.to_dict(orient='records')

    # Makale detaylarını (başlık, abstract, kategori vb.) İÇERMEYEN hafif hover
    hover_texts = [
        f"ID: {row['external_id']}<br>Risk: {float(row['risk_skoru']):.3f}<br>Küme: {row['kume']}"
        for row in records
    ]

    # 1. Ana Trace: Tüm makaleler (minimum customdata ile)
    main_trace = go.Scattergl(
        x=plot_df['umap_x'].tolist(),
        y=plot_df['umap_y'].tolist(),
        mode='markers',
        customdata=records,
        marker=dict(
            size=6,
            color=plot_df['risk_skoru'].tolist(),
            colorscale=[
                [0, '#474747'],
                [0.5, "#8F5D5D"],
                [1, "#bb0000"],
            ],
            showscale=True,
            colorbar=dict(title='Risk', thickness=10, len=0.8),
            opacity=0.8,
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Makaleler'
    )

    # 2. Seçim Trace'i: Tıklanan nokta burada öne çıkar (başlangıçta boştur)
    selected_trace = go.Scattergl(
        x=[],
        y=[],
        mode='markers',
        marker=dict(
            size=12,               # Boyutu büyük
            color="#3167A5",       # Doğrudan rengi değişmiş hali
            opacity=1.0
        ),
        hoverinfo='skip',
        name='Seçilen'
    )

    # İki trace'i birden grafiğe veriyoruz
    fig = go.Figure(data=[main_trace, selected_trace])

    fig.update_layout(
        title=dict(text=''),
        showlegend=False,
        margin=dict(l=25, r=20, t=35, b=25),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8fafc',
        xaxis=dict(title='UMAP 1', gridcolor='#e2e8f0', zeroline=False),
        yaxis=dict(title='UMAP 2', gridcolor='#e2e8f0', zeroline=False),
        hovermode='closest',
        dragmode='pan',  
    )

  fig_dict = fig.to_dict()
  fig_dict['config'] = {'scrollZoom': True, 'displayModeBar': True}

  graph_json = json.dumps(fig_dict, cls=plotly.utils.PlotlyJSONEncoder)
  return Response(graph_json, mimetype='application/json')


@app.route('/api/article/<external_id>', methods=['GET'])
def get_article(external_id):
  try:
    article = load_article_detail(external_id)
    if article is None:
      return jsonify({'error': 'Makale bulunamadı.'}), 404
    return jsonify(article)
  except Exception as e:
    return jsonify({'error': str(e)}), 500


#Küme merkezlerini ve etiketlerini front-end'e sunan route
@app.route('/api/cluster-summaries', methods=['GET'])
def get_cluster_summaries():
    """
    HDBSCAN küme özetlerini (merkezler, etiketler, boyutlar) front-end'e JSON olarak sunar.
    """
    summary_path = 'data/hdbscan_cluster_summary.csv'
    
    if not os.path.exists(summary_path):
        return jsonify({"error": "Küme özet dosyası henüz oluşturulmamış. Lütfen pipeline'ı çalıştırın."}), 404
        
    try:
        df_summary = pd.read_csv(summary_path)
        # NaN değerleri temizle (JSON serileştirme hatası vermemesi için)
        df_summary = df_summary.fillna("")
        
        # DataFrame'i dictionary listesine çevir
        summaries = df_summary.to_dict(orient='records')
        return jsonify(summaries)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/evaluation', methods=['GET'])
def get_evaluation():
  res = {
      'seeded_kmeans': [],
      'baseline_kmeans': [],
      'embedding_comparison': [],
  }

  seeded_path = os.path.join(RESULTS_DIR, 'kmeans', 'seeded_evaluation_summary.csv')
  if os.path.exists(seeded_path):
    try:
      res['seeded_kmeans'] = pd.read_csv(seeded_path).to_dict(orient='records')
    except Exception:
      pass

  baseline_path = os.path.join(RESULTS_DIR, 'kmeans', 'baseline_evaluation_summary.csv')
  if os.path.exists(baseline_path):
    try:
      res['baseline_kmeans'] = pd.read_csv(baseline_path).to_dict(orient='records')
    except Exception:
      pass

  emb_path = os.path.join(RESULTS_DIR, 'embedding_model_comparison.csv')
  if os.path.exists(emb_path):
    try:
      res['embedding_comparison'] = pd.read_csv(emb_path).to_dict(orient='records')
    except Exception:
      pass

  return jsonify(res)



# ==============================================================================
# K-MEANS FINAL KONU TAHMİN SAYFASI
# HDBSCAN route ve veri akışına dokunmadan eklenmiştir.
# ==============================================================================

def _km_text(value):
  if pd.isna(value):
    return ''
  return str(value).strip()


def _km_topics(value):
  text = _km_text(value)
  if not text:
    return []
  return [part.strip() for part in text.split('||') if part.strip()]


def _load_final_kmeans_records():
  # Öncelik: Adaptive V2. Dosya yoksa eski final Relabel çıktısına düşer.
  prediction_candidates = [
      os.path.join(RESULTS_DIR, 'kmeans', 'holdout', 'adaptive_v2_relabel_predictions.csv'),
      os.path.join(RESULTS_DIR, 'kmeans', 'holdout', 'relabel_kmeans_predictions.csv'),
  ]

  prediction_path = next((path for path in prediction_candidates if os.path.exists(path)), None)
  article_path = os.path.join(BASE_DIR, 'data', 'balanced_articles.csv')

  if prediction_path is None or not os.path.exists(article_path):
    return [], {}

  pred_df = pd.read_csv(prediction_path, dtype={'external_id': str}, encoding='utf-8-sig')
  article_df = pd.read_csv(article_path, dtype={'external_id': str}, encoding='utf-8-sig', low_memory=False)

  pred_df['external_id'] = pred_df['external_id'].astype(str).str.strip()
  article_df['external_id'] = article_df['external_id'].astype(str).str.strip()

  meta_cols = [
      col for col in [
          'external_id', 'doi', 'publication_year', 'publication_type',
          'language', 'title', 'abstract', 'keywords'
      ] if col in article_df.columns
  ]

  merged = pred_df.merge(article_df[meta_cols], on='external_id', how='left')
  records = []

  for _, row in merged.iterrows():
    predicted = _km_topics(row.get('full_topic_predictions', ''))
    true_topics = _km_topics(row.get('true_subjects', ''))

    pred_set = set(predicted)
    true_set = set(true_topics)
    matched = sorted(pred_set & true_set)
    wrong = sorted(pred_set - true_set)
    missed = sorted(true_set - pred_set)

    tp = len(matched)
    fp = len(wrong)
    fn = len(missed)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    match_rate = tp / len(true_set) if true_set else 0.0

    record = {
        'external_id': _km_text(row.get('external_id', '')),
        'title': _km_text(row.get('title', '')),
        'doi': _km_text(row.get('doi', '')),
        'year': _km_text(row.get('publication_year', '')),
        'publication_type': _km_text(row.get('publication_type', '')),
        'language': _km_text(row.get('language', '')),
        'abstract': _km_text(row.get('abstract', '')),
        'keywords': _km_text(row.get('keywords', '')),
        'main_topics': _km_topics(row.get('main_level_2_predictions', '')),
        'leaf_topics': _km_topics(row.get('leaf_predictions', '')),
        'predicted_topics': predicted,
        'true_topics': true_topics,
        'matched': matched,
        'wrong': wrong,
        'missed': missed,
        'matched_count': tp,
        'wrong_count': fp,
        'missed_count': fn,
        'true_count': len(true_set),
        'predicted_count': len(pred_set),
        'match_rate': match_rate,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    records.append(record)

  return records, {record['external_id']: record for record in records}


def _load_final_kmeans_summary():
  # V2 varsa onu göster. Yoksa eski final karşılaştırmasına düş.
  comparison_candidates = [
      os.path.join(RESULTS_DIR, 'kmeans', 'holdout', 'adaptive_v2_fixed_vs_relabel_comparison.csv'),
      os.path.join(RESULTS_DIR, 'kmeans', 'holdout', 'fixed_vs_relabel_comparison.csv'),
  ]

  defaults = {
      'micro_precision': 0.3525,
      'micro_recall': 0.3527,
      'micro_f1': 0.3526,
      'macro_f1': 0.3166,
      'sample_f1': 0.3665,
      'exact_match': 0.0513,
      'at_least_one': 0.6874,
      'average_labels': 2.29,
      'unique_topics': 178,
      'version': 'Adaptive V2',
  }

  path = next((path for path in comparison_candidates if os.path.exists(path)), None)
  if not path:
    return defaults

  try:
    comp = pd.read_csv(path, encoding='utf-8-sig')
    relabel = comp[comp['Model'].astype(str).str.contains('Relabel', case=False, na=False)]
    row = relabel.iloc[0] if not relabel.empty else comp.iloc[-1]
    return {
        'micro_precision': float(row.get('Micro_Precision', defaults['micro_precision'])),
        'micro_recall': float(row.get('Micro_Recall', defaults['micro_recall'])),
        'micro_f1': float(row.get('Micro_F1', defaults['micro_f1'])),
        'macro_f1': float(row.get('Macro_F1', defaults['macro_f1'])),
        'sample_f1': float(row.get('Sample_F1', defaults['sample_f1'])),
        'exact_match': float(row.get('Exact_Match_Rate', defaults['exact_match'])),
        'at_least_one': float(row.get('At_Least_One_Match_Rate', defaults['at_least_one'])),
        'average_labels': float(row.get('Average_Predicted_Labels', defaults['average_labels'])),
        'unique_topics': int(row.get('Unique_Topics', defaults['unique_topics'])),
        'version': 'Adaptive V2' if 'adaptive_v2' in path else 'Final K-Means',
    }
  except Exception:
    return defaults


@app.route('/kmeans')
def kmeans_page():
  return render_template('kmeans.html', summary=_load_final_kmeans_summary())


@app.route('/api/kmeans/articles', methods=['GET'])
def get_kmeans_articles():
  records, _ = _load_final_kmeans_records()

  query = request.args.get('search', '').strip().lower()
  match_filter = request.args.get('match', 'ALL')
  pred_count = request.args.get('pred_count', 'ALL')
  sort_key = request.args.get('sort', 'match_desc')

  filtered = records

  if query:
    filtered = [
        row for row in filtered
        if query in row['external_id'].lower()
        or query in row['title'].lower()
        or query in row['doi'].lower()
        or any(query in topic.lower() for topic in row['true_topics'])
        or any(query in topic.lower() for topic in row['predicted_topics'])
    ]

  if match_filter == 'ZERO':
    filtered = [row for row in filtered if row['matched_count'] == 0]
  elif match_filter == 'ONE':
    filtered = [row for row in filtered if row['matched_count'] == 1]
  elif match_filter == 'TWO':
    filtered = [row for row in filtered if row['matched_count'] == 2]
  elif match_filter == 'THREE_PLUS':
    filtered = [row for row in filtered if row['matched_count'] >= 3]
  elif match_filter == 'WRONG_ANY':
    filtered = [row for row in filtered if row['wrong_count'] > 0]
  elif match_filter == 'WRONG_NONE':
    filtered = [row for row in filtered if row['wrong_count'] == 0 and row['predicted_count'] > 0]
  elif match_filter == 'RATE_100':
    filtered = [row for row in filtered if row['true_count'] > 0 and row['match_rate'] == 1.0]
  elif match_filter == 'RATE_50_99':
    filtered = [row for row in filtered if 0.5 <= row['match_rate'] < 1.0]

  if pred_count in {'1', '2', '3'}:
    filtered = [row for row in filtered if row['predicted_count'] == int(pred_count)]

  if sort_key == 'match_asc':
    filtered = sorted(filtered, key=lambda row: (row['match_rate'], row['f1']))
  elif sort_key == 'f1_desc':
    filtered = sorted(filtered, key=lambda row: row['f1'], reverse=True)
  elif sort_key == 'f1_asc':
    filtered = sorted(filtered, key=lambda row: row['f1'])
  else:
    filtered = sorted(filtered, key=lambda row: (row['match_rate'], row['f1']), reverse=True)

  payload = [
      {
          'external_id': row['external_id'],
          'title': row['title'],
          'year': row['year'],
          'matched_count': row['matched_count'],
          'wrong_count': row['wrong_count'],
          'true_count': row['true_count'],
          'predicted_count': row['predicted_count'],
          'match_rate': row['match_rate'],
          'precision': row['precision'],
          'recall': row['recall'],
          'f1': row['f1'],
      }
      for row in filtered[:500]
  ]

  return jsonify({'data': payload, 'total': len(filtered)})


@app.route('/api/kmeans/article/<external_id>', methods=['GET'])
def get_kmeans_article(external_id):
  _, record_map = _load_final_kmeans_records()
  record = record_map.get(str(external_id).strip())
  if record is None:
    return jsonify({'error': 'Makale bulunamadı.'}), 404
  return jsonify(record)


@app.route('/kmeans/umap')
def kmeans_umap():
  umap_path = os.path.join(
      RESULTS_DIR,
      'kmeans',
      'kmeans_v2_umap_interactive.html'
  )

  if not os.path.exists(umap_path):
    return (
        '<h3>K-Means UMAP grafiği bulunamadı.</h3>'
        '<p>Önce clustering/kmeans/generate_kmeans_v2_umap.py dosyasını çalıştırın.</p>',
        404
    )

  return send_file(umap_path)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5001, debug=True)