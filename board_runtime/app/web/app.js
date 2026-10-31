(()=>{'use strict';
const $=id=>document.getElementById(id);let token=document.querySelector('meta[name=home-token]').content;
let current=null,pending=0,pendingSince=0,busy=false,toastTimer,reconnectTimer,incidentSignature='',incidentBusy=false;
const stateNames={IDLE:'监测中',CANDIDATE:'正在确认',ALARM_ARMED:'正在确认',ALARM:'检测到倒地',STARTING:'启动中',STOPPED:'已停止'};
for(let i=0;i<8;i++){const button=document.createElement('button');button.className='led-button';button.setAttribute('aria-pressed','false');button.setAttribute('aria-label',`切换 LED ${i}`);button.innerHTML=`<span class="led-lamp"></span><span>LED ${i}</span>`;button.onclick=()=>command({led:{index:i,on:!(current&&current.controls.applied.led_mask&(1<<i))}});$('led-grid').appendChild(button);}
function toast(text,error=false){clearTimeout(toastTimer);$('toast').textContent=text;$('toast').className='toast show'+(error?' error':'');toastTimer=setTimeout(()=>$('toast').classList.remove('show'),3000);}
function age(seconds){if(seconds<60)return `${seconds} 秒前`;if(seconds<3600)return `${Math.floor(seconds/60)} 分钟前`;return `${Math.floor(seconds/3600)} 小时前`;}
function duration(seconds){if(seconds<60)return `${Math.floor(seconds)} 秒`;if(seconds<3600)return `${Math.floor(seconds/60)} 分钟`;return `${Math.floor(seconds/3600)} 小时 ${Math.floor(seconds%3600/60)} 分钟`;}
function enableControls(enabled){document.querySelectorAll('.controls-card button,.controls-card input,.controls-card select').forEach(e=>e.disabled=!enabled||busy);if(!(current&&current.controls.image&&current.controls.image.available))document.querySelectorAll('.image-card button,.image-card input').forEach(e=>e.disabled=true);if(current&&current.controls.semantic&&current.controls.semantic.mode==='live')$('replay-rotation').disabled=true;if(current&&recognitionMode(current.detector,current.controls)!=='gesture'&&recognitionMode(current.detector,current.controls)!=='both')$('gesture-enabled').disabled=true;}
let imageEditing=false,imageTimer=null,imageQueued=null;
function queueImage(){imageQueued={brightness:Number($('image-brightness').value),contrast:Number($('image-contrast').value)};clearTimeout(imageTimer);imageTimer=setTimeout(flushImage,350);}
async function flushImage(){if(!imageQueued)return;if(busy){imageTimer=setTimeout(flushImage,150);return;}const value=imageQueued;imageQueued=null;imageEditing=false;await command({image:value});}
function renderImage(image,d,connected){
 const im=image||{},available=connected&&im.available;
 $('image-status').textContent=available?'PL 已就绪':'等待硬件';$('image-status').className='hardware-badge'+(available?' ready':'');
 document.querySelectorAll('[data-image]').forEach(e=>e.classList.toggle('selected',e.dataset.image===im.mode));
 if(!imageEditing&&!busy&&!pending){$('image-brightness').value=im.brightness||0;$('image-contrast').value=im.contrast||100;$('brightness-value').textContent=im.brightness||0;$('contrast-value').textContent=(im.contrast||100)+'%';}
 const pl=d.pl_statistics;$('inference-engine').textContent=!connected?'等待推理状态':d.engine==='ncnn_pl'?`推理：PS＋PL · 累计 ${pl?pl.calls:0} 次硬件卷积`:'推理：PS CPU';
}
const gestureNames={open_palm:'张开手掌',fist:'握拳',victory:'V 手势',unknown:'手势未确认'};
const handHints={no_palm:'未找到完整手掌，请让整只手入镜',uncertain_landmarks:'手部关键点不清晰，请稳定姿势',hand_clipped:'手掌超出画面，请稍微远离镜头',too_small:'手部太小，请适当靠近',unknown_pose:'手势未确认'};
function recognitionMode(d,c){
 const selected=(c.semantic||{}).recognition_mode||d.recognition_mode||'both';
 return d.recognition_mode&&d.recognition_mode!==selected?'paused':selected;
}
function recognitionLabel(mode){return {fall:'摔倒模式',gesture:'手势模式',paused:'识别暂停'}[mode]||'现场识别';}
function renderSemantic(d,c,connected){
 const s=c.semantic||{};$('semantic-card').hidden=!s.available;if(!s.available)return;
 const rm=recognitionMode(d,c),stale=rm!=='gesture'&&typeof d.evidence_age_s==='number'&&d.evidence_age_s>3,changing=rm==='paused'||d.monitor_mode!==s.mode;
 $('recognition-mode').textContent=recognitionLabel(rm)+' · '+((s.recognition_switch||{}).message||'由开发板控制');
 $('semantic-status').textContent=!connected?'未连接':changing?'切换中':stale?'结果未更新':'已就绪';
 $('semantic-status').className='hardware-badge'+(connected&&!changing&&!stale?' ready':'');
 document.querySelectorAll('[data-monitor]').forEach(e=>e.classList.toggle('selected',e.dataset.monitor===s.mode));
 $('gesture-enabled').checked=s.gesture_enabled;if(document.activeElement!==$('replay-rotation'))$('replay-rotation').value=String(s.rotation||0);
 const reviewed=(d.reviewed_candidates||[]).length;
 $('semantic-summary').textContent=!connected?'等待开发板连接':rm==='gesture'?'手势识别运行中，摔倒识别与报警暂停':rm==='paused'?'拨码切换中或接口未就绪，识别暂停':d.coverage_limited?'目标较多，部分候选尚待复核':stale?'识别结果暂未更新，请确认现场情况':`本次已复核 ${reviewed} 个非倒地候选 · 结果约 ${Number(d.evidence_age_s||0).toFixed(1)} 秒前`;
 const hand=d.hand,gesture=d.gesture_control||{},fresh=typeof d.hand_age_s==='number'&&d.hand_age_s<=3;
 $('gesture-name').textContent=!connected?'手势状态未连接':rm==='fall'?'摔倒模式中，手势暂停':rm==='paused'?'识别暂停':!s.gesture_enabled?'手势控制已关闭':s.mode!=='live'?'回放测试中，手势控制暂停':hand&&fresh&&hand.multiple_hands?'检测到多只手，请只保留一只手入镜':hand&&fresh?(gestureNames[hand.name]||'手势未确认'):(handHints[d.hand_status]||'等待完整单手入镜');
 $('gesture-action').textContent=gesture.last_action?'最近执行：'+gesture.last_action+(typeof gesture.last_action_age_s==='number'?` · ${Math.floor(gesture.last_action_age_s)} 秒前`:''):'尚未执行手势控制';
}
function renderEnvironment(value,connected){
 const e=value||{},valid=connected&&e.valid;
 $('environment-temperature').textContent=valid?Number(e.temperature_c).toFixed(1):'—';
 $('environment-humidity').textContent=valid?Number(e.humidity_rh).toFixed(1):'—';
 $('environment-status').textContent=!connected?'未连接':valid?'已更新':e.available?'等待有效读数':'等待数据';
 $('environment-status').className='hardware-badge'+(valid?' ready':'');
 $('environment-message').textContent=!connected?'开发板未连接，环境读数暂不可用':e.message||'等待开发板环境读数';
 $('environment-age').textContent=connected&&e.age_s!==null&&e.age_s!==undefined?`最近有效读数：${duration(e.age_s)}前`:'温湿度约每5秒更新';
}
async function post(path,payload){const controller=typeof AbortController!=='undefined'?new AbortController():null,options={method:'POST',headers:{'Content-Type':'application/json','X-Home-Token':token},body:JSON.stringify(payload)};if(controller)options.signal=controller.signal;const timer=controller?setTimeout(()=>controller.abort(),6000):null;try{const response=await fetch(path,options),body=await response.json();if(!response.ok)throw Error(body.error||'请求失败');return body;}finally{clearTimeout(timer);}}
async function command(payload){if(!current||busy)return;busy=true;enableControls(true);try{const body=await post('/api/control',payload);pending=body.revision;pendingSince=Date.now();toast('控制已提交，等待板端执行');await poll();}catch(error){toast(error.name==='AbortError'?'请求超时，请检查开发板连接':error.message,true);}finally{busy=false;enableControls(Boolean(current&&current.detector.running));}}

function renderVideoHud(d,c,connected){
 const stage=$('video-stage'),e=current.environment||{},valid=connected&&e.valid,rm=recognitionMode(d,c);
 stage.classList.toggle('is-alarm',connected&&(rm==='fall'||rm==='both')&&d.state_name==='ALARM');
 $('video-state').textContent=!connected?'未连接':rm==='gesture'||rm==='paused'?recognitionLabel(rm):stateNames[d.state_name]||'等待识别';
 $('video-score').textContent=connected&&rm!=='gesture'&&rm!=='paused'?Math.max(0,Math.min(1,Number(d.pe)||0)).toFixed(2):'—';
 $('test-context').textContent=d.test_context==='phone_screen_replay'?'手机回放测试':d.test_context==='ui_preview'?'界面预览':'现场画面';
 $('video-temperature').textContent=valid?Number(e.temperature_c).toFixed(1):'—';
 $('video-humidity').textContent=valid?Number(e.humidity_rh).toFixed(1):'—';
 const s=c.semantic||{},hand=d.hand,fresh=typeof d.hand_age_s==='number'&&d.hand_age_s<=3;
 $('video-gesture').textContent=!connected?'等待连接':rm==='fall'?'手势已暂停':rm==='paused'?'识别暂停':s.gesture_enabled===false?'手势已关闭':s.mode==='phone_replay'?'回放中暂停':hand&&fresh?(gestureNames[hand.name]||'手势未确认'):'等待手势';
 $('video-gesture-hint').textContent=!connected?'检查板端程序':rm==='fall'?'SW0 向上切换手势':rm==='paused'?'检查 SW0 与 FPGA 配置':s.mode==='phone_replay'?'手机回放联调':handHints[d.hand_status]||'保持姿势，等待稳定确认';
}
function updateVideoStats(stats,server={}){
 const fps=stats.native?server.fps:stats.fps;
 $('web-fps').textContent=fps?fps.toFixed(1):'—';
 $('stream-info').textContent=fps?`${fps.toFixed(1)} FPS · ${stats.native?'视频流':'最新帧'}`:'等待视频流';
}
const videoPlayer=new HomeVideo.VideoPlayer($('video-canvas'),$('camera'),stats=>{
 $('video-wait').hidden=true;
 $('video-resolution').textContent=`${stats.width} × ${stats.height}`;
 updateVideoStats(stats);
},message=>{ $('camera-link').textContent=message; });
function render(data){current=data;token=data.control_token||token;const d=data.detector,c=data.controls,v=data.video,alarm=d.state_name==='ALARM'&&['fall','both'].includes(recognitionMode(d,c)),ready=d.ready&&d.running;
 const rm=recognitionMode(d,c),fallActive=rm==='fall'||rm==='both';
 const board=d.running&&data.status_age_s<3,clientVideo=videoPlayer.stats(),fresh=clientVideo.native?board&&v.frame_age_s!==null&&v.frame_age_s<3:clientVideo.fresh;
 renderEnvironment(data.environment,board);
 renderImage(c.image,d,board);
 renderSemantic(d,c,board);
 renderVideoHud(d,c,board);
 $('connection').className='connection '+(board?'online':'offline');$('connection').innerHTML='<i></i>'+(board?'已连接':'开发板未就绪');
 $('gateway-link').textContent='网页服务：已连接';$('board-link').textContent=board?'开发板：已连接':'开发板：未连接或已停止';$('camera-link').textContent=fresh?'相机：实时画面':'相机：画面未更新';
 $('state-text').textContent=fallActive?(stateNames[d.state_name]||'监测中'):recognitionLabel(rm);$('state-text').style.color=alarm?'#c53d3d':'#2c7052';
 $('state-description').textContent=!d.running?'识别程序已停止':rm==='gesture'?'摔倒识别与报警暂停':rm==='paused'?'等待拨码稳定或更新 FPGA 配置':!d.ready?'模型正在启动':alarm?'请及时确认现场情况':d.state_name==='IDLE'?'未发现倒地目标':'正在确认倒地状态';
 $('safety-card').classList.toggle('alarm',alarm&&board);$('safety-title').textContent=!board?'识别未连接':!fallActive?recognitionLabel(rm):alarm?'请及时查看':ready?'正在守护':'系统启动中';
 $('safety-subtitle').textContent=!board?'请检查板端程序、供电与网线':rm==='gesture'?'SW0 向下可恢复摔倒识别':rm==='paused'?'等待拨码稳定，请检查 SW0 接口':alarm?'检测到持续倒地，请确认现场情况':ready?'摄像头与识别系统正在运行':'等待摄像头与模型就绪';
 if(fallActive&&d.coverage_limited&&board&&!alarm){$('safety-title').textContent='覆盖范围受限';$('safety-subtitle').textContent='目标较多，部分候选尚未完成复核';}
 if(fallActive&&typeof d.evidence_age_s==='number'&&d.evidence_age_s>3&&board&&!alarm){$('safety-title').textContent='识别结果未更新';$('safety-subtitle').textContent='视频仍可观看，请确认现场情况';}
 const score=fallActive?Math.max(0,Math.min(1,Number(d.pe)||0)):0;$('fall-score').textContent=fallActive?score.toFixed(2):'—';$('score-bar').style.width=`${score*100}%`;
 $('web-fps').textContent=clientVideo.native?(v.fps?v.fps.toFixed(1):'—'):(clientVideo.fps?clientVideo.fps.toFixed(1):'—');$('hdmi-fps').textContent=`HDMI ${d.display_fps?d.display_fps.toFixed(1):'—'} FPS`;
 $('alarm-count').textContent=c.alarm_count;$('uptime').textContent=`本次运行 ${duration(d.uptime_s||0)}`;
 $('video-resolution').textContent=Array.isArray(v.resolution)?v.resolution.join(' × '):'等待画面';$('viewer-info').textContent=`${v.viewers} 台设备正在观看`;updateVideoStats(clientVideo,v);
 $('inference-info').textContent=!fallActive?'当前未运行摔倒识别':d.inference_ms?`识别 ${Math.round(d.inference_ms)} ms / 次 · ${d.inference_fps?d.inference_fps.toFixed(1):'—'} 次 / 秒`:'评分不代表识别准确率';
 $('video-wait').hidden=fresh;
 if(!fresh)$('video-wait').querySelector('strong').textContent=v.sequence?'画面暂未更新':'正在连接摄像头';
 document.querySelectorAll('.led-button').forEach((e,i)=>e.setAttribute('aria-pressed',String(Boolean(c.applied.led_mask&(1<<i)))));
 document.querySelectorAll('[data-ps]').forEach(e=>e.classList.toggle('selected',e.dataset.ps===c.desired.ps_led));
 $('led-mode').textContent=c.desired.led_auto?'自动模式：LED 0 指示摄像头已就绪。':'手动模式：点击上方 LED 可独立开关。';
 $('buzzer-state').className='sound-state'+(c.applied.buzzer?' active':'');$('buzzer-state').innerHTML=`<i></i>${c.applied.buzzer?'正在鸣叫':'静音'}`;
 $('buzzer-auto').checked=c.desired.buzzer_auto;$('mute-note').textContent=c.muted_seconds?`自动短鸣静默中，剩余 ${c.muted_seconds} 秒。`:'手动停止后，自动短鸣静默 60 秒。';
 const applied=c.applied_revision>=c.revision;$('control-state').textContent=!board?'板端离线':applied?'已同步':'执行中';$('control-state').className='hardware-badge'+(applied&&board?' ready':'');
 if(pending&&c.applied_revision>=pending){pending=0;toast('板端已执行，设备状态已同步');}
 if(pending&&Date.now()-pendingSince>8000){pending=0;toast('未收到板端执行回执，请检查连接',true);}
 const events=$('events');if(c.events.length){events.replaceChildren();for(const event of c.events.slice(0,5)){const row=document.createElement('div');row.className='event '+event.kind;const icon=document.createElement('span');icon.className='event-icon';icon.textContent=event.kind==='alarm'?'!':'↗';const text=document.createElement('span');text.textContent=event.message;const time=document.createElement('time');time.textContent=age(event.age_s);row.append(icon,text,time);events.append(row);}}
 enableControls(Boolean(board));if(!c.applied.ps_led_available)document.querySelectorAll('[data-ps]').forEach(e=>e.disabled=true);
}
let pollBusy=false;
async function poll(){if(pollBusy)return;pollBusy=true;const controller=typeof AbortController!=='undefined'?new AbortController():null;const timer=controller?setTimeout(()=>controller.abort(),4000):null;try{const options={cache:'no-store'};if(controller)options.signal=controller.signal;const response=await fetch('/api/state',options);if(!response.ok)throw Error('offline');render(await response.json());}catch(e){current=null;renderEnvironment(null,false);$('connection').className='connection offline';$('connection').innerHTML='<i></i>连接中断';$('gateway-link').textContent='网页服务：连接中断';$('board-link').textContent='开发板：状态未知';$('camera-link').textContent='相机：画面未更新';$('safety-title').textContent='连接中断';$('safety-subtitle').textContent='请检查网络连接与设备运行状态';$('state-text').textContent='连接中断';enableControls(false);$('video-wait').hidden=false;$('video-wait').querySelector('strong').textContent='正在重新连接';}finally{pollBusy=false;clearTimeout(timer);}}
const contexts={live_scene:'现场画面',phone_screen_replay:'手机回放测试',synthetic_output_test:'联动测试',synthetic:'模拟测试',ui_preview:'界面预览'};
const reviews={unreviewed:'待确认',confirmed:'已确认倒地',false_alarm:'已标记误报'};
async function incidents(force=false){if(incidentBusy)return;incidentBusy=true;try{const response=await fetch('/api/incidents',{cache:'no-store'});if(!response.ok)throw Error('记录暂不可用');const data=await response.json(),signature=JSON.stringify(data);if(!force&&signature===incidentSignature)return;
 if($('incidents').contains(document.activeElement)&&!force)return;incidentSignature=signature;const box=$('incidents');box.textContent='';if(!data.available||!data.items.length){const p=document.createElement('p');p.className='muted';p.textContent=data.available?'尚无报警记录':'报警记录暂不可用';box.appendChild(p);return;}
 for(const item of data.items){const row=document.createElement('article');row.className='incident';if(item.snapshot){const image=document.createElement('img');image.src='/api/incidents/'+item.id+'/snapshot.jpg';image.alt='报警时截图';image.loading='lazy';row.appendChild(image);}const content=document.createElement('div');content.className='incident-content';const title=document.createElement('strong');title.textContent=new Date(item.started).toLocaleString();const info=document.createElement('p');info.textContent=(contexts[item.context]||'来源未标记')+' · '+reviews[item.review]+' · '+(item.ended?'已解除':'待核实结束时间');const note=document.createElement('input');note.type='text';note.maxLength=300;note.placeholder='备注，例如：坐下、衣物误报';note.value=item.note||'';note.setAttribute('aria-label','报警备注');const actions=document.createElement('div');actions.className='incident-actions';for(const value of ['confirmed','false_alarm','unreviewed']){const button=document.createElement('button');button.className='button secondary';button.textContent=reviews[value];button.onclick=async()=>{button.disabled=true;try{const result=await fetch('/api/incidents/'+item.id+'/review',{method:'POST',headers:{'Content-Type':'application/json','X-Home-Token':token},body:JSON.stringify({review:value,note:note.value})});const body=await result.json();if(!result.ok)throw Error(body.error||'保存失败');toast('反馈已保存');incidentSignature='';await incidents(true);}catch(error){toast(error.message,true);}finally{button.disabled=false;}};actions.appendChild(button);}content.append(title,info,note,actions);row.appendChild(content);box.appendChild(row);}
 }catch(error){if(!incidentSignature)$('incidents').textContent='记录暂不可用，请稍后刷新';}finally{incidentBusy=false;}}
$('refresh-incidents').onclick=()=>incidents(true);
$('all-on').onclick=()=>command({led_mask:255});$('all-off').onclick=()=>command({led_mask:0});$('led-auto').onclick=()=>command({led_auto:true});
document.querySelectorAll('[data-ps]').forEach(e=>e.onclick=()=>command({ps_led:e.dataset.ps}));
$('beep-on').onclick=()=>command({buzzer:{on:true,duration_ms:2000}});$('beep-off').onclick=()=>command({buzzer:{on:false}});$('buzzer-auto').onchange=e=>command({buzzer_auto:e.target.checked});
document.querySelectorAll('[data-image]').forEach(e=>e.onclick=()=>command({image:{mode:e.dataset.image}}));
$('image-brightness').oninput=e=>{imageEditing=true;$('brightness-value').textContent=e.target.value;queueImage();};
$('image-contrast').oninput=e=>{imageEditing=true;$('contrast-value').textContent=e.target.value+'%';queueImage();};
$('image-apply').onclick=()=>{queueImage();clearTimeout(imageTimer);flushImage();};
$('image-reset').onclick=()=>{clearTimeout(imageTimer);imageQueued=null;imageEditing=false;command({image:{mode:'color',brightness:0,contrast:100}});};
document.querySelectorAll('[data-monitor]').forEach(e=>e.onclick=()=>command({semantic:{mode:e.dataset.monitor}}));
$('gesture-enabled').onchange=e=>command({semantic:{gesture_enabled:e.target.checked}});
$('replay-rotation').onchange=e=>command({semantic:{rotation:Number(e.target.value)}});
$('fullscreen').onclick=()=>{const target=$('video-wrap');if(document.fullscreenElement)document.exitFullscreen();else if(target.requestFullscreen)target.requestFullscreen().catch(()=>toast('当前浏览器不支持全屏'));};

function clock(){const d=new Date(),pad=n=>n<10?'0'+n:String(n);$('clock').textContent=pad(d.getMonth()+1)+'/'+pad(d.getDate())+' '+pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds());}
enableControls(false);clock();setInterval(clock,1000);poll();setInterval(poll,500);incidents();setInterval(incidents,5000);
videoPlayer.start();
document.addEventListener('visibilitychange',()=>{if(document.hidden)videoPlayer.stop();else{videoPlayer.start();poll();}});
window.addEventListener('pagehide',()=>videoPlayer.stop());
})();
