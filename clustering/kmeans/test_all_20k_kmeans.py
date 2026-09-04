import os, runpy
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from config.paths import INDEX_FILE

BASE_SCRIPT='clustering/kmeans/adaptive_label_count_v2_20k.py'
ARTICLE_FILE='data/balanced_articles.csv'
SUBJECT_FILE='data/article_subjects.csv'
OUTPUT_DIR='results/kmeans'

print('='*110)
print('20.902 MAKALE - TÜM VERİ K-MEANS KONU TAHMİN TESTİ')
print('='*110)
print('\n[1/6] Mevcut Adaptive V2 modeli çalıştırılıyor (orijinal dosya değiştirilmiyor)...')
g=runpy.run_path(BASE_SCRIPT, run_name='__all20k_base__')

create_distance_matrix=g['create_distance_matrix']
adaptive_predictions=g['adaptive_predictions']
best_second_ratio=float(g['best_second_ratio'])
best_third_ratio=float(g['best_third_ratio'])
centroid_subjects=list(g['centroid_subjects'])
leaf_subjects=list(g['leaf_subjects'])
relabel_subjects=list(g.get('relabel_subjects', centroid_subjects))

print('Second ratio:', best_second_ratio)
print('Third ratio :', best_third_ratio)

articles=pd.read_csv(ARTICLE_FILE,encoding='utf-8-sig',dtype={'external_id':str},low_memory=False)
index_df=pd.read_csv(INDEX_FILE,encoding='utf-8-sig',dtype={'external_id':str})
subjects=pd.read_csv(SUBJECT_FILE,encoding='utf-8-sig',dtype={'external_id':str})
for df in (articles,index_df,subjects):
    df['external_id']=df['external_id'].astype(str).str.strip()

all_df=(articles.merge(index_df[['external_id','embedding_row']],on='external_id',how='inner')
        .drop_duplicates('external_id').reset_index(drop=True))
print('\n[2/6] Makale:',len(articles),'Embedding ile eşleşen:',len(all_df))

print('\n[3/6] Tüm makaleler için centroid uzaklıkları...')
all_distances=create_distance_matrix(all_df)
print('Distance matrix:',all_distances.shape)

print('\n[4/6] Adaptive V2 tahminleri...')
y_cluster_pred,predicted_counts=adaptive_predictions(all_distances,best_second_ratio,best_third_ratio)
print('1 etiket:',int((predicted_counts==1).sum()))
print('2 etiket:',int((predicted_counts==2).sum()))
print('3 etiket:',int((predicted_counts==3).sum()))
print('Ort. tahmin:',round(float(predicted_counts.mean()),3))

subject_to_index={s:i for i,s in enumerate(leaf_subjects)}
true_subject_map=(subjects.groupby('external_id')['subject_fullname']
    .apply(lambda s: sorted(set(x.strip() for x in s.dropna().astype(str) if x.strip()))).to_dict())
y_true=np.zeros((len(all_df),len(leaf_subjects)),dtype=np.int8)
for i,eid in enumerate(all_df['external_id']):
    for s in true_subject_map.get(eid,[]):
        j=subject_to_index.get(s)
        if j is not None: y_true[i,j]=1

def cluster_to_leaf(cluster_pred, cluster_subjects):
    out=np.zeros_like(y_true)
    for cid,subj in enumerate(cluster_subjects):
        j=subject_to_index.get(subj)
        if j is not None:
            out[cluster_pred[:,cid]==1,j]=1
    return out

y_fixed=cluster_to_leaf(y_cluster_pred,centroid_subjects)
y_relabel=cluster_to_leaf(y_cluster_pred,relabel_subjects)

def metrics(name,pred):
    return {
      'Model':name,'Articles':len(all_df),
      'Micro_Precision':precision_score(y_true,pred,average='micro',zero_division=0),
      'Micro_Recall':recall_score(y_true,pred,average='micro',zero_division=0),
      'Micro_F1':f1_score(y_true,pred,average='micro',zero_division=0),
      'Macro_F1':f1_score(y_true,pred,average='macro',zero_division=0),
      'Sample_F1':f1_score(y_true,pred,average='samples',zero_division=0),
      'Exact_Match_Rate':float(np.all(y_true==pred,axis=1).mean()),
      'At_Least_One_Match_Rate':float((((y_true & pred).sum(axis=1))>0).mean()),
      'Average_Predicted_Labels':float(pred.sum(axis=1).mean()),
      'Average_True_Labels':float(y_true.sum(axis=1).mean())}

fixed_m=metrics('Sabit Konu Kimligi',y_fixed)
relabel_m=metrics('Relabel - Baskin Konu',y_relabel)

def levels(topics):
    l1=[]; l2=[]; leaves=[]
    for full in topics:
        p=[x.strip() for x in full.split('>') if x.strip()]
        if p: l1.append(p[0]); leaves.append(p[-1])
        if len(p)>=2: l2.append(' > '.join(p[:2]))
    return list(dict.fromkeys(l1)),list(dict.fromkeys(l2)),list(dict.fromkeys(leaves))

def pred_df(pred):
    rows=[]
    for i,row in all_df.iterrows():
        selected=np.where(pred[i]==1)[0]
        full=[leaf_subjects[j] for j in selected]
        l1,l2,leaves=levels(full)
        eid=row['external_id']
        rows.append({'external_id':eid,'predicted_label_count':len(full),
          'main_level_1_predictions':' || '.join(l1),'main_level_2_predictions':' || '.join(l2),
          'leaf_predictions':' || '.join(leaves),'full_topic_predictions':' || '.join(full),
          'true_subjects':' || '.join(true_subject_map.get(eid,[]))})
    return pd.DataFrame(rows)

os.makedirs(OUTPUT_DIR,exist_ok=True)
pred_df(y_fixed).to_csv(os.path.join(OUTPUT_DIR,'all_20k_fixed_predictions.csv'),index=False,encoding='utf-8-sig')
pred_df(y_relabel).to_csv(os.path.join(OUTPUT_DIR,'all_20k_relabel_predictions.csv'),index=False,encoding='utf-8-sig')
pd.DataFrame([fixed_m,relabel_m]).to_csv(os.path.join(OUTPUT_DIR,'all_20k_evaluation_summary.csv'),index=False,encoding='utf-8-sig')

print('\n'+'='*110)
print('TÜM 20.902 MAKALE - KONU ÜRETME / EŞLEŞME SONUÇLARI')
print('='*110)
for r in (fixed_m,relabel_m):
    print('\n'+r['Model'])
    print('-'*70)
    print(f"Micro Precision: {r['Micro_Precision']*100:.2f}%")
    print(f"Micro Recall:    {r['Micro_Recall']*100:.2f}%")
    print(f"Micro F1:        {r['Micro_F1']*100:.2f}%")
    print(f"Macro F1:        {r['Macro_F1']*100:.2f}%")
    print(f"Sample F1:       {r['Sample_F1']*100:.2f}%")
    print(f"Exact Match:     {r['Exact_Match_Rate']*100:.2f}%")
    print(f"En az 1 doğru:   {r['At_Least_One_Match_Rate']*100:.2f}%")
    print(f"Ort. tahmin:     {r['Average_Predicted_Labels']:.2f}")
    print(f"Ort. gerçek:     {r['Average_True_Labels']:.2f}")

print('\nDosyalar:')
print('results/kmeans/all_20k_fixed_predictions.csv')
print('results/kmeans/all_20k_relabel_predictions.csv')
print('results/kmeans/all_20k_evaluation_summary.csv')