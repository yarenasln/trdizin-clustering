const $=id=>document.getElementById(id); const pct=v=>`${(Number(v)*100).toFixed(1)}%`;
function esc(s){return String(s??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[m]))}
async function loadArticles(){
 const p=new URLSearchParams({q:$("q").value,match:$("match").value,pred_count:$("predCount").value,sort:$("sort").value});
 const res=await fetch(`/api/kmeans/articles?${p}`); const data=await res.json();
 $("countText").textContent=`${data.total} makale bulundu`;
 $("articleList").innerHTML=data.articles.map(a=>`<div class="article-item" data-id="${esc(a.external_id)}">
 <div class="article-id">${esc(a.external_id)} ${a.year?`• ${esc(a.year)}`:""}</div>
 <div class="article-title">${esc(a.title||"Başlık bilgisi yok")}</div>
 <div class="row-stats"><span class="match-chip">${a.matched_count}/${a.true_count} konu • ${pct(a.match_rate)}</span><span class="f1-chip">F1 ${pct(a.f1)}</span></div>
 <div class="mini-bar"><i style="width:${Math.min(100,a.match_rate*100)}%"></i></div></div>`).join("");
 document.querySelectorAll(".article-item").forEach(el=>el.onclick=()=>loadArticle(el.dataset.id,el));
}
function renderTags(arr,kind=""){return (arr&&arr.length)?arr.map(x=>`<span class="tag ${kind}">${esc(x)}</span>`).join(""):`<span class="tag">Yok</span>`}
async function loadArticle(id,el){
 document.querySelectorAll(".article-item").forEach(x=>x.classList.remove("active")); if(el)el.classList.add("active");
 const res=await fetch(`/api/kmeans/article/${encodeURIComponent(id)}`); const a=await res.json(); if(!res.ok)return alert(a.error);
 $("detail").className="detail";
 $("detail").innerHTML=`<span class="kicker">MAKALE DETAYI</span><h2>${esc(a.title||"Başlık bilgisi yok")}</h2>
 <div class="meta"><span>ID: ${esc(a.external_id)}</span><span>DOI: ${esc(a.doi||"Yok")}</span><span>Yıl: ${esc(a.year||"Yok")}</span><span>Dil: ${esc(a.language||"Yok")}</span></div>
 <div class="section-title">ÖZET</div><div class="abstract">${esc(a.abstract||"Özet bilgisi yok")}</div>
 <div class="section-title">KONU KARŞILAŞTIRMASI</div>
 <div class="topic-grid"><div class="topic-box"><h4>Gerçek TR Dizin Konuları</h4><div class="tags">${renderTags(a.true_topics)}</div></div>
 <div class="topic-box"><h4>K-Means Tahminleri</h4><div class="tags">${renderTags(a.matched,"good")}${renderTags(a.wrong,"bad")}</div></div></div>
 <div class="topic-grid" style="margin-top:12px"><div class="topic-box"><h4>✓ Doğru Yakalanan</h4><div class="tags">${renderTags(a.matched,"good")}</div></div>
 <div class="topic-box"><h4>! Kaçırılan</h4><div class="tags">${renderTags(a.missed,"missed")}</div></div></div>
 <div class="match-panel"><div class="match-top"><div><b>${a.matched_count} / ${a.true_count} gerçek konu yakalandı</b><div style="color:#bdc8d7;margin-top:4px">Gerçek konuların yakalanma oranı</div></div><div class="match-percent">${pct(a.match_rate)}</div></div>
 <div class="progress"><i style="width:${Math.min(100,a.match_rate*100)}%"></i></div>
 <div class="article-metrics"><div><span>PRECISION</span><strong>${pct(a.precision)}</strong></div><div><span>RECALL</span><strong>${pct(a.recall)}</strong></div><div><span>F1</span><strong>${pct(a.f1)}</strong></div></div></div>`;
}
$("searchBtn").onclick=loadArticles; $("q").addEventListener("keydown",e=>{if(e.key==="Enter")loadArticles()});
["match","predCount","sort"].forEach(id=>$(id).onchange=loadArticles); loadArticles();