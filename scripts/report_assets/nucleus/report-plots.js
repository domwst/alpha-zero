'use strict';
const DATA=JSON.parse(document.getElementById('audit-data').textContent), $=id=>document.getElementById(id), nf=new Intl.NumberFormat('en-US');
const fmt=(x,d=1)=>Number(x).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}), pct=(x,d=2)=>fmt(100*x,d)+'%';
const fields=['support_before','support_after','removed_count','removed_fraction','removed_mass'];
const thresholds=['0.9','0.95','0.98','0.99','1.0'];
const visible={support:[true,true],comparison:thresholds.map(()=>true)};
let activeGroup=null,filterCache=null,resetRevision=0,renderSequence=0;

function merged(p,stage='all',seat='all'){
 const out={positions:0,sum_before:0,sum_removed:0,sum_fraction:0,sum_mass:0,observed:0,excluded:0};
 fields.forEach(k=>out[k]=Array(Object.values(DATA.thresholds[p])[0][k].length).fill(0));
 const joint=new Map();
 for(const [key,g] of Object.entries(DATA.thresholds[p])){
  const [s,t]=key.split('/');if((stage!=='all'&&stage!==s)||(seat!=='all'&&seat!==t))continue;
  for(const k in out){if(Array.isArray(out[k]))g[k].forEach((v,i)=>out[k][i]+=v);else out[k]+=g[k]}
  for(const [before,after,n]of g.support_joint){const key=before*362+after;joint.set(key,(joint.get(key)||0)+n)}
 }
 out.support_joint=Array.from(joint,([key,n])=>[Math.floor(key/362),key%362,n]);
 return out;
}
function quantile(hist,q){const n=hist.reduce((a,b)=>a+b,0),index=Math.max(0,Math.ceil(n*q)-1);let c=0;for(let i=0;i<hist.length;i++){c+=hist[i];if(c>index)return i}return 0}
function median(hist){const n=hist.reduce((a,b)=>a+b,0);const at=index=>{let c=0;for(let i=0;i<hist.length;i++){c+=hist[i];if(c>index)return i}return 0};return (at(Math.floor((n-1)/2))+at(Math.floor(n/2)))/2}
function color(token){const e=document.createElement('span');e.style.color=`var(${token})`;document.body.append(e);const c=getComputedStyle(e).color;e.remove();return c}
function palette(){return{surface:color('--ds-color-surface'),text:color('--ds-color-text'),muted:color('--ds-color-text-muted'),border:color('--ds-color-border'),neutral:color('--ds-color-neutral'),series:[1,2,3,4].map(n=>color(`--ds-chart-series-${n}`)),font:getComputedStyle(document.documentElement).getPropertyValue('--ds-font-ui').trim(),dark:!matchMedia('print').matches&&(document.documentElement.dataset.theme==='dark'||(!document.documentElement.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches))}}
const config={responsive:true,displaylogo:false,displayModeBar:true,scrollZoom:false,showTips:false,editable:false,doubleClick:'reset',modeBarButtonsToRemove:['select2d','lasso2d','sendChartToCloud'],toImageButtonOptions:{format:'png',scale:2,filename:'gomoku-nucleus-sampling'}};
function baseLayout(node,p,revision){
 const narrow=node.clientWidth<420;
 return{autosize:true,height:node.clientHeight,font:{family:p.font,size:12,color:p.text},paper_bgcolor:p.surface,plot_bgcolor:p.surface,margin:{l:narrow?48:58,r:16,t:38,b:54},showlegend:false,dragmode:'zoom',hovermode:'x unified',hoverlabel:{bgcolor:p.surface,bordercolor:p.border,font:{family:p.font,color:p.text,size:12}},uirevision:revision,transition:{duration:0},
  xaxis:{gridcolor:p.border,zeroline:false,showspikes:true,spikemode:'across',spikesnap:'cursor',spikecolor:p.muted,spikethickness:1,automargin:true,nticks:narrow?4:6},
  yaxis:{gridcolor:p.border,zeroline:false,automargin:true,nticks:5},modebar:{bgcolor:p.surface,color:p.muted,activecolor:p.series[0]}};
}
function controls(node){
 if(!node._auditControlsBound){node.on('plotly_relayout',()=>controls(node));node._auditControlsBound=true}
 node.querySelectorAll('.modebar-btn').forEach(button=>{
  button.setAttribute('role','button');button.setAttribute('tabindex','0');button.setAttribute('aria-label',button.getAttribute('data-title')||button.getAttribute('aria-label')||'Chart control');
  if(button.dataset.attr==='dragmode')button.setAttribute('aria-pressed',String(button.classList.contains('active')));
  button.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();button.click();controls(node)}};
 });
}
function legend(id,names,colors){
 const host=$(id).querySelector('.legend');host.replaceChildren();
 names.forEach((name,i)=>{const b=document.createElement('button');b.type='button';b.setAttribute('aria-pressed',String(visible[id][i]));const swatch=document.createElement('i');swatch.className='swatch';swatch.style.background=colors[i];b.append(swatch,document.createTextNode(name));b.onclick=()=>{visible[id][i]=!visible[id][i];b.setAttribute('aria-pressed',String(visible[id][i]));const node=$(id).querySelector('.plot');Plotly.restyle(node,{visible:visible[id][i]?true:'legendonly'},[i])};host.append(b)});
}
const descriptors=[{id:'support',keys:['support_before','support_after'],names:['Before','Retained'],step:1,unit:'moves'}, {id:'removed',keys:['removed_count'],names:['Removed'],step:1,unit:'moves'}, {id:'fraction',keys:['removed_fraction'],names:['Positions'],step:.001,unit:'fraction'}, {id:'mass',keys:['removed_mass'],names:['Positions'],step:.0001,unit:'fraction'}];

async function draw(desc,g,p){
 const node=$(desc.id).querySelector('.plot'),mode=$('mode').value,res=$('resolution').value;
 const bin=desc.step===1?({fine:1,standard:5,coarse:20}[res]):({fine:1,standard:10,coarse:50}[res]);
 const domain=desc.id==='mass'?Math.min(1001,Math.max(101,Math.ceil((1-Number($('threshold').value))*10000)+1)):g[desc.keys[0]].length;
 const hist=desc.keys.map(key=>{const a=[];g[key].slice(0,domain).forEach((v,i)=>{const j=Math.floor(i/bin);a[j]=(a[j]||0)+v});return a});
 const colors=desc.id==='support'?[p.neutral,p.series[0]]:[desc.id==='mass'?p.series[2]:p.series[0]];
 const x=[],width=[],labels=[],upper=[];
 for(let i=0;i<hist[0].length;i++){
  const lo=i*bin*desc.step,hi=Math.min((i+1)*bin,domain)*desc.step,mult=desc.unit==='moves'?1:100;
  const last=desc.unit==='moves'?hi-1:hi;x.push((lo+last)/2*mult);width.push((hi-lo)*mult);upper.push(last*mult);
  labels.push(desc.unit==='moves'?(lo===last?`${lo} moves`:`${lo}–${last} moves`):`${pct(lo,desc.id==='mass'?2:1)} to <${pct(hi,desc.id==='mass'?2:1)}`);
 }
 const values=hist.map(a=>{if(mode==='hist')return a;let c=0;return a.map(v=>(c+=v)/g.positions*100)});
 const traces=values.map((y,i)=>({type:mode==='hist'?'bar':'scatter',mode:'lines',name:desc.names[i],x:mode==='hist'?x:upper,y,width:mode==='hist'?width.map(w=>w*.9):undefined,
  customdata:y.map((v,j)=>[labels[j],hist[i][j],hist[i][j]/g.positions*100]),
  hovertemplate:mode==='hist'?'%{customdata[0]}<br>%{y:,} positions (%{customdata[2]:.2f}%)<extra>%{fullData.name}</extra>':'Through %{customdata[0]}<br>%{y:.2f}% of positions<extra>%{fullData.name}</extra>',
  marker:{color:colors[i],opacity:p.dark?.7:1,line:desc.id==='support'&&i===0?{color:p.neutral,width:1}:undefined},
  line:{color:colors[i],width:2,dash:desc.id==='support'&&i===0?'dash':'solid',shape:'hv'},visible:visible[desc.id]?.[i]===false?'legendonly':true}));
 const revision=['one-d',desc.id,$('threshold').value,$('stage').value,$('seat').value,mode,res,resetRevision].join('/');
 const layout=baseLayout(node,p,revision);layout.barmode='group';layout.bargap=.05;
 layout.xaxis.title={text:desc.unit==='moves'?'Number of moves':desc.id==='mass'?'Probability mass removed (%)':'Nonzero moves removed (%)'};
 layout.xaxis.range=desc.unit==='moves'?[-.5,361.5]:[0,(domain-1)*desc.step*100];
 layout.yaxis.title={text:mode==='hist'?'Positions':'Cumulative positions (%)'};layout.yaxis.rangemode='tozero';if(mode==='cdf')layout.yaxis.range=[0,101];
 await Plotly.react(node,traces,layout,config);Plotly.Fx.unhover(node);controls(node);
 if(desc.id==='support')legend('support',desc.names,colors);
 const inspector=$(desc.id).querySelector('.inspect');let selected=0;
 function inspect(index){selected=Math.max(0,Math.min(labels.length-1,index));inspector.textContent=labels[selected]+' · '+values.map((a,i)=>`${desc.names[i]}: ${mode==='hist'?nf.format(a[selected])+' positions ('+pct(a[selected]/g.positions)+')':fmt(a[selected],2)+'%'}`).join(' · ')}
 // Inspect the whole horizontal bin, even when its bar is only a pixel high.
 node.onpointermove=e=>{const rect=node.getBoundingClientRect(),axis=node._fullLayout.xaxis;const value=axis.p2d(e.clientX-rect.left-axis._offset);inspect(Math.floor(value/(desc.unit==='moves'?bin:bin*desc.step*100)))};
 node.onkeydown=e=>{if(e.target!==node)return;if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();inspect(e.key==='Home'?0:e.key==='End'?labels.length-1:selected+(e.key==='ArrowRight'?1:-1));Plotly.Fx.hover(node,[{curveNumber:0,pointNumber:selected}])}};
 inspect(0);
}

function jointBins(g,bin){
 const n=Math.ceil(361/bin),matrix=Array.from({length:n},()=>Array(n).fill(0));
 for(const [b,a,count]of g.support_joint)matrix[Math.floor((a-1)/bin)][Math.floor((b-1)/bin)]+=count;
 return matrix;
}
async function drawJoint(groups,p){
 const top=$('threshold').value,g=groups[top],bin={fine:1,standard:5,coarse:20}[$('resolution').value],node=$('joint').querySelector('.plot');
 const matrix=jointBins(g,bin),max=Math.max(...Object.values(groups).map(group=>Math.max(...jointBins(group,bin).map(row=>Math.max(...row)))));
 const n=matrix.length,x=Array.from({length:n},(_,i)=>(i*bin+1+Math.min((i+1)*bin,361))/2),bounds=Array.from({length:n},(_,i)=>[i*bin+1,Math.min((i+1)*bin,361)]);
 const custom=matrix.map((row,y)=>row.map((count,x)=>[...bounds[x],...bounds[y],count,count/g.positions*100]));
 const z=matrix.map(row=>row.map(v=>v?Math.log10(v):null)),ticks=Array.from({length:Math.floor(Math.log10(Math.max(1,max)))+1},(_,i)=>i);
 const trace={type:'heatmap',x,y:x,z,zmin:0,zmax:Math.max(1,Math.log10(max)),customdata:custom,hoverongaps:false,zsmooth:false,
  colorscale:[[0,p.neutral],[1,p.series[0]]],colorbar:{title:{text:'Positions<br>(log color)'},tickvals:ticks,ticktext:ticks.map(t=>nf.format(10**t)),thickness:12,len:.8,tickfont:{size:12,color:p.muted},outlinewidth:0},
  hovertemplate:'Before: %{customdata[0]}–%{customdata[1]} moves<br>Retained: %{customdata[2]}–%{customdata[3]} moves<br>%{customdata[4]:,} positions (%{customdata[5]:.3f}%)<extra>Top-p '+top+'</extra>'};
 const layout=baseLayout(node,p,['joint',$('stage').value,$('seat').value,$('resolution').value,resetRevision].join('/'));
 const narrow=node.clientWidth<420;layout.margin.r=12;layout.margin.b=110;layout.hovermode='closest';
 const colorTicks=ticks.filter((v,i)=>i%2===0||i===ticks.length-1);
 Object.assign(trace.colorbar,{orientation:'h',x:.5,xanchor:'center',y:-.27,yanchor:'top',len:.95,thickness:10,title:{text:'Positions (log color)',side:'top'},tickvals:colorTicks,ticktext:colorTicks.map(t=>new Intl.NumberFormat('en-US',{notation:'compact'}).format(10**t))});
 layout.xaxis={...layout.xaxis,title:{text:'Nonzero moves before',standoff:8},tickangle:0,nticks:narrow?3:5,range:[.5,361.5],constrain:'domain'};
 layout.yaxis={...layout.yaxis,title:{text:'Nonzero moves retained',standoff:4},nticks:narrow?3:5,range:[.5,361.5],scaleanchor:'x',scaleratio:1,constrain:'domain'};
 layout.shapes=[{type:'line',x0:1,y0:1,x1:361,y1:361,line:{color:p.neutral,width:1.5,dash:'dash'}}];
 await Plotly.react(node,[trace],layout,config);Plotly.Fx.unhover(node);controls(node);
 $('joint-reading').textContent=`Reading: top-p ${top}, ${bin} × ${bin} move-count bins. ${nf.format(g.positions)} positions contribute to this heatmap; ${pct(g.removed_count[0]/g.positions)} retain exactly their original support. The dashed diagonal marks no change. Below it, filtering removed moves. Use Fine for exact integer pairs.`;
}
function meanByBefore(g){const sums=Array(362).fill(0);for(const [b,a,n]of g.support_joint)sums[b]+=a*n;return sums.map((v,b)=>g.support_before[b]?v/g.support_before[b]:null)}
async function drawComparison(groups,p){
 const node=$('comparison').querySelector('.plot'),colors=[...p.series,p.neutral],x=Array.from({length:361},(_,i)=>i+1);
 const traces=thresholds.map((top,i)=>{const g=groups[top],means=meanByBefore(g);return{type:'scatter',mode:'lines',x,y:means.slice(1),name:'Top-p '+top+(top==='1.0'?' · no filtering':''),line:{color:colors[i],width:top===$('threshold').value?3:1.8,dash:top==='1.0'?'dash':'solid'},connectgaps:false,customdata:x.map(b=>g.support_before[b]),visible:visible.comparison[i]?true:'legendonly',hovertemplate:'Before: %{x} moves<br>Mean retained: %{y:.2f}<br>%{customdata:,} positions<extra>Top-p '+top+'</extra>'}});
 const layout=baseLayout(node,p,['comparison',$('stage').value,$('seat').value,resetRevision].join('/'));
 layout.xaxis={...layout.xaxis,title:{text:'Nonzero moves before'},range:[1,361]};layout.yaxis={...layout.yaxis,title:{text:'Mean moves retained'},range:[0,361]};
 await Plotly.react(node,traces,layout,config);Plotly.Fx.unhover(node);controls(node);legend('comparison',traces.map(t=>t.name),colors);
 const g=groups['1.0'],before=g.support_before[361]?361:g.support_before.findLastIndex(n=>n>0);
 $('comparison-reading').textContent=`Reading: among ${nf.format(g.support_before[before])} positions with ${before} original nonzero moves, mean retained moves are ${thresholds.map(top=>`${fmt(meanByBefore(groups[top])[before])} at top-p ${top}`).join('; ')}. Stage and seat filters apply to every curve. The selected top-p is drawn thicker.`;
}

async function render(){
 const sequence=++renderSequence,p=palette(),key=$('stage').value+'/'+$('seat').value;
 if(filterCache?.key!==key)filterCache={key,groups:Object.fromEntries(thresholds.map(t=>[t,merged(t,$('stage').value,$('seat').value)]))};
 const groups=filterCache.groups,g=groups[$('threshold').value];activeGroup=g;
 $('view-context').textContent=`Top-p ${$('threshold').value} · ${$('stage').selectedOptions[0].text} · ${$('seat').selectedOptions[0].text} · ${nf.format(g.positions)} of ${nf.format(DATA.positions)} positions · ${$('mode').selectedOptions[0].text}, ${$('resolution').value} bins`;
 $('view-summary').innerHTML=`<span><b>${fmt(g.sum_removed/g.positions)}</b> moves removed / position</span><span><b>${pct(g.sum_fraction/g.positions)}</b> position-average fraction removed</span><span><b>${pct(g.sum_removed/g.sum_before)}</b> of all nonzero entries removed</span><span><b>${pct(g.sum_mass/g.positions)}</b> mean mass removed</span>`;
 $('support-reading').textContent=`Reading: mean nonzero moves ${fmt(g.sum_before/g.positions)} → ${fmt((g.sum_before-g.sum_removed)/g.positions)}. Median ${fmt(median(g.support_before))} → ${fmt(median(g.support_after))}; ${pct(g.support_after[1]/g.positions)} of positions retain exactly one move.`;
 $('removed-reading').textContent=`Reading: ${nf.format(g.sum_removed)} nonzero position–move entries removed in this view; mean ${fmt(g.sum_removed/g.positions)}, median ${fmt(median(g.removed_count))}, 90th percentile ${quantile(g.removed_count,.9)} moves per position. ${pct(g.removed_count[0]/g.positions)} of positions lose no moves.`;
 $('fraction-reading').textContent=`Reading: the position-average fraction removed is ${pct(g.sum_fraction/g.positions)}. Across all nonzero position–move entries in this view, ${pct(g.sum_removed/g.sum_before)} are removed. The chart gives every position equal weight.`;
 $('mass-reading').textContent=`Reading: mean probability mass removed is ${pct(g.sum_mass/g.positions)}. ${nf.format(g.excluded)} of ${nf.format(g.observed)} recoverable played actions (${pct(g.excluded/g.observed)}) fall outside the retained set. This does not label them as mistakes.`;
 await Promise.all([...descriptors.map(d=>draw(d,g,p)),drawJoint(groups,p),drawComparison(groups,p)]);
 if(sequence===renderSequence)document.body.dataset.ready='true';
}
function redraw(){document.body.dataset.ready='false';render().catch(error=>{console.error(error);$('view-context').textContent='Chart rendering failed: '+error.message})}
['threshold','stage','seat','mode','resolution'].forEach(id=>$(id).addEventListener('change',redraw));
$('reset').onclick=()=>{for(const [id,v]of Object.entries({threshold:'0.95',stage:'all',seat:'all',mode:'hist',resolution:'standard'}))$(id).value=v;visible.support=[true,true];visible.comparison=thresholds.map(()=>true);resetRevision++;redraw()};
$('download').onclick=()=>{const blob=new Blob([JSON.stringify(DATA)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='nucleus-replay-distributions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
(()=>{const root=document.documentElement,picker=document.querySelector('.theme-picker'),KEY='nucleus-audit-theme';let choice=root.dataset.theme||'system';const sync=()=>{if(choice==='system')delete root.dataset.theme;else root.dataset.theme=choice;picker.querySelectorAll('input').forEach(i=>i.checked=i.value===choice)};picker.addEventListener('change',e=>{choice=e.target.value;sync();try{if(choice==='system')localStorage.removeItem(KEY);else localStorage.setItem(KEY,choice)}catch{}redraw()});addEventListener('pageshow',sync);matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{if(choice==='system')redraw()});sync()})();
let resizeTimer;addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(redraw,180)});
addEventListener('beforeprint',redraw);addEventListener('afterprint',redraw);
redraw();
