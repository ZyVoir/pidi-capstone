import J from"jquery";import{p as T}from"./purify.es-DKgZpLtg.js";import{S as Et}from"./shepherd-Bpa11e5q.js";import{i as Ae,S as Me,r as ie}from"./markdown-renderer-DiuhccNt.js";import{_ as Be}from"./preload-helper-BXl3LOEh.js";import{c as Oe,n as Xt,g as yt,a as vt,b as Yt,d as qe}from"./index-DgnXmQO9.js";import"./_commonjsHelpers-CE1G-McA.js";var et=(t=>(t.ACTIVE="active",t.MOVED="moved",t.ORPHANED="orphaned",t))(et||{});function K(t){return t.replaceAll(/\u00A0/g," ").replaceAll(/\n/g," ")}function Jt(t,e,a=0){if(typeof t=="string"){const s=t.indexOf(e,a);if(s===-1)throw new Error("Searched string not found");return s}else{const s=t.indexOf(e,a);if(s===-1)throw new Error("Searched item not found in array");return s}}function re(t,e){const a=K(e.textContent??""),s=K(t.toString());let u,l;const h=document.createTreeWalker(e,NodeFilter.SHOW_TEXT);let f=null,k=null,v=[];for(;h.nextNode()&&!k;)h.currentNode===t.startContainer&&(f=h.currentNode),f||(v=v.concat(h.currentNode)),h.currentNode===t.endContainer&&(k=h.currentNode);u=v.reduce((N,b)=>N+K(b.textContent??""),"").length+t.startOffset,l=u+s.length;const A=30,R=a.slice(Math.max(0,u-A),u),w=a.slice(l,l+A);return{highlightedText:s,highlightStartOffset:u,highlightEndOffset:l,highlightPrefix:R,highlightSuffix:w}}function Re(t){return t.replace(/_([a-z])/g,(e,a)=>a.toUpperCase())}function ut(t){return Array.isArray(t)?t.map(e=>ut(e)):t!==null&&typeof t=="object"?Object.fromEntries(Object.entries(t).map(([e,a])=>[Re(e),ut(a)])):t}function $e(t,e){return K(e.textContent??"").slice(t.highlightStartOffset,t.highlightEndOffset)===t.highlightedText?{...t,status:et.ACTIVE}:{...t,status:et.ORPHANED}}function Kt(t,e){const a=e.highlightStartOffset,s=e.highlightEndOffset,u=document.createTreeWalker(t,NodeFilter.SHOW_TEXT),l=document.createRange();let h=0,f=null,k=0,v=null,j=0;for(;u.nextNode();){const w=u.currentNode,b=K(w.textContent??"").length;if(!f&&h+b>=a&&(f=w,k=a-h),!v&&h+b>=s){v=w,j=s-h;break}h+=b}if(!f||!v)throw new Error("Offset out of bounds");l.setStart(f,k),l.setEnd(v,j);function A(w){return w.parentElement===null?null:w.parentElement===t?w:A(w.parentElement)}function R(w){return w.nodeType===Node.ELEMENT_NODE&&w.childNodes.length?R(w.childNodes[0]):w}if(l.startContainer.textContent?.length===l.startOffset){const w=l.endContainer.parentElement?.closest("p"),N=l.startContainer.parentElement?.closest("li"),b=l.endContainer.parentElement?.closest("li");if(b)N===b?l.setStart(l.endContainer,0):l.setStart(b,0);else if(w)l.setStart(w,0);else{const x=A(l.startContainer.parentElement),at=x.nextElementSibling.childNodes;at.length?l.setStart(R(at[0]),0):l.setStart(x.nextElementSibling,0)}}if(l.startContainer.nodeType===Node.ELEMENT_NODE&&l.startContainer.tagName.toLowerCase()==="img"){const w=A(l.startContainer.parentElement),N=w.nextElementSibling.childNodes;N.length?l.setStart(R(N[0]),0):l.setStart(w.nextElementSibling,0)}return l}async function ae(t){return await(await fetch(window._sharedData.storeNewLearningNoteEndpoint,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(t)})).json()}function Qt(t,e){const s=document.querySelector("#learning-note-indicators-item-button-template").content.cloneNode(!0);return s.firstElementChild?.setAttribute("data-note-id",t.id.toString()),s.firstElementChild?.append(t.highlightedText.length>30?`${t.highlightedText.substring(0,30)}...`:t.highlightedText),s.firstElementChild?.addEventListener("click",u=>{const l=u.currentTarget,h=Number(l.dataset.noteId);h<1||isNaN(h)||e(h)}),s}const De="/api/v1/ai-explanation",D="source:karsa",te=10,Pe=180,_={LOADING:"Sedang berpikir...",MAX_LENGTH_ERROR:"Mohon untuk memilih kalimat yang lebih pendek untuk dijelaskan.",GENERAL_ERROR:"Maaf, output tidak bisa dihasilkan. Silakan coba kembali."};function Fe(t){const e=t.getRangeAt(0);return{firstRect:e.getClientRects()[0],activeRange:e,highlightedText:t?.toString()?.trim()??""}}function He(t){const e=document.querySelector("#tutorial-content").getBoundingClientRect(),a=Math.abs(e.top);return a>80?a+t.firstRect.bottom+t.firstRect.height+6*te:t.firstRect.top+t.firstRect.height+te}function Ct(t){try{const e=new Date(t);if(isNaN(e.getTime()))throw new Error("Date format invalid");const a=new Date,s=new Date(a.getFullYear(),a.getMonth(),a.getDate()),u=new Date(s.getTime()+1440*60*1e3),l=new Date(e.getFullYear(),e.getMonth(),e.getDate()),h={hour:"2-digit",minute:"2-digit",hour12:!1},f=new Intl.DateTimeFormat("id-ID",h).format(e);if(l.getTime()===s.getTime())return f;if(l.getTime()===u.getTime())return`besok pukul ${f}`;const k={day:"numeric",month:"short",hour:"2-digit",minute:"2-digit",hour12:!1};return new Intl.DateTimeFormat("id-ID",k).format(e)}catch(e){return e instanceof Error?e.message:"An error occurred"}}class Ge{freemiumModal=null;premiumModal=null;constructor(){this.initializeModals()}initializeModals(){this.freemiumModal=document.getElementById("explain-ai-limit-freemium-modal"),this.premiumModal=document.getElementById("explain-ai-limit-premium-modal"),this.setupFreemiumModalHandlers(),this.setupPremiumModalHandlers()}setupFreemiumModalHandlers(){if(!this.freemiumModal)return;const e=this.freemiumModal.querySelector("#freemium-modal-close"),a=this.freemiumModal.querySelector("#freemium-modal-close-btn"),s=this.freemiumModal.querySelector("#freemium-modal-upgrade-btn"),u=()=>this.hideFreemiumModal();e?.addEventListener("click",u),a?.addEventListener("click",u),s?.addEventListener("click",()=>{this.handleUpgrade()}),this.freemiumModal.addEventListener("click",l=>{l.target===this.freemiumModal&&u()})}setupPremiumModalHandlers(){if(!this.premiumModal)return;const e=this.premiumModal.querySelector("#premium-modal-close"),a=this.premiumModal.querySelector("#premium-modal-ok-btn"),s=()=>this.hidePremiumModal();e?.addEventListener("click",s),a?.addEventListener("click",s),this.premiumModal.addEventListener("click",u=>{u.target===this.premiumModal&&s()})}showLimitModal(e){this.hideAllModals(),e.isPremiumUser?this.showPremiumModal(e.resetTime):this.showFreemiumModal()}showPremiumModal(e){if(this.premiumModal){if(e){const a=this.premiumModal.querySelector("#premium-reset-time");a&&(a.textContent=e)}this.premiumModal.style.display="flex",document.body.style.overflow="hidden"}}showFreemiumModal(){this.freemiumModal&&(this.freemiumModal.style.display="flex",document.body.style.overflow="hidden")}hidePremiumModal(){this.premiumModal&&(this.premiumModal.style.display="none",document.body.style.overflow="")}hideFreemiumModal(){this.freemiumModal&&(this.freemiumModal.style.display="none",document.body.style.overflow="")}handleUpgrade(){const e="/subscriptions/purchases";window.location.href=e}hideAllModals(){this.hidePremiumModal(),this.hideFreemiumModal()}}const Ve=new Ge;async function Ue(t,e,a){const s=t.body?.getReader(),u=new TextDecoder;if(s)try{for(;;){const{done:l,value:h}=await s.read();if(l){a&&a();break}const f=u.decode(h,{stream:!0});e(f)}}catch(l){console.error("Streaming error:",l)}finally{s.releaseLock()}}async function se(t,e){try{if(e(_.LOADING,!1),t.highlightedText.length>Pe)throw new RangeError;const a=await fetch(De,{method:"POST",headers:{"Content-Type":"application/json","x-csrf-token":window.Dicoding.csrfToken},body:JSON.stringify({course_id:t.courseId,highlighted_text:t.highlightedText})});if(e("",!1),!a.ok){const l=await a.json();throw new Error(l.message)}let s="";return await Ue(a,l=>{s+=l,e(l,!0)},async()=>{const{completeStreaming:l}=await Be(async()=>{const{completeStreaming:h}=await Promise.resolve().then(()=>dn);return{completeStreaming:h}},void 0);l()}),!s.includes("Mohon maaf, tiba - tiba terdapat error dalam sistem AI.")}catch(a){const s=a instanceof RangeError?_.MAX_LENGTH_ERROR:_.GENERAL_ERROR;throw e(s,!1),a}}async function We(t){try{if(!t.noteBody||t.noteBody.trim().length===0)throw new Error("noteBody is required and cannot be empty");if(t.noteBody===_.LOADING)throw new Error("Cannot save note while explanation is still loading");if(!t.highlightedText||t.highlightedText.trim().length===0)throw new Error("No text selected to create a note");if(!t.notesData)throw new Error("Note data is not available");const e={_token:t.token,note_body:`${D} ${t.noteBody.trim()}`,highlighted_text:t.notesData.highlightedText,highlight_start_offset:t.notesData.highlightStartOffset.toString(),highlight_end_offset:t.notesData.highlightEndOffset.toString(),highlight_prefix:t.notesData.highlightPrefix,highlight_suffix:t.notesData.highlightSuffix};return{success:!0,message:"note saved successfully",data:await ae(e)}}catch(e){return{success:!1,message:e instanceof Error?e.message:"An error occurred while saving the note"}}}let d=null,U=null,ht=!1,nt=!1,_t=null,St=null,P=null,F=null,S=null,B=null,W=null,ot=null,y=null,Q=null,tt=null,I=null,O=null,z=!1,gt=!1;function ze(){U?.removeAttribute("disabled")}function je(){he(S?.limit===S?.used,`${S?.used}/${S?.limit}`,Ct(`${S?.reset?.nextResetAt}`))}function Ze(t){z=t,t&&(Lt(),ge(),q(),it(_.LOADING))}async function Xe(t){if(!t)return Promise.resolve(!1);const e=t;if(ot=e,S?.remaining===0)return fe(),G(),H(),!1;try{const a=await se(e,it);return a?(q(),gt=!0):rt(),a}catch(a){return console.error(a),a instanceof RangeError?q():rt(),!1}}function Ye(t){S=t,nt=O.mode==="subscription"||O.mode==="internal",window.explainAiRemainingLimit={limit:t.limit,mode:O?.mode},an(nt),sn(t.remaining,t.limit)}function Je(t){console.warn("Feature limit reached! Remaining : "+t.used+" / "+t.limit)}function le(t,e){d=e,U=t,_t=e.querySelector("#annotateBtn"),St=e.querySelector("#saveNoteBtn"),P=e.querySelector("#fallbackCloseBtn"),F=e.querySelector("#fallbackRetryBtn"),me(),tn(),ce(),de(),U&&U.addEventListener("click",a=>{if(!nn())return a.preventDefault(),a.stopPropagation(),!1},!0),Ke()}function mt(){return O!==null||U===null||(O=Ae("explain-with-ai",{onFeatureBeingUsed:Ze,onFeatureDisabled:je,onFeatureEnabled:ze,onUseFeature:Xe,onApproachingLimits:Je,onRemainingLimitsUpdated:Ye})),O}function Ke(){const t=document.querySelector(".learning-notes__eai-credit-info");if(!t)return;const e=new IntersectionObserver(a=>{a.some(s=>s.isIntersecting)&&(e.disconnect(),mt())});e.observe(t)}function ce(){d&&(P&&(P.removeEventListener("click",bt),P.addEventListener("click",bt)),F&&(F.removeEventListener("click",xt),F.addEventListener("click",xt)))}function bt(t){t.preventDefault(),t.stopPropagation(),G()}function xt(t){t.preventDefault(),t.stopPropagation(),ue()}function de(){d&&(H(),Q=t=>{t.stopPropagation();const e=t.target;if(z&&!e.classList.contains("popup-close")){t.preventDefault(),t.stopImmediatePropagation();return}if(e.classList.contains("popup-close")){H(),G();return}if(e===_t){Qe();return}if(e===St){en().catch(a=>{console.error("Save failed:",a)});return}},tt=t=>{const e=t.target;if(z&&d&&!d.contains(e))return t.preventDefault(),t.stopImmediatePropagation(),!1;if(!(e.closest("#learning-note-creation-backdrop")||e.closest("#user-selection-floating-buttons-backdrop")||e.closest(".sidebar-navigation-btn-show")||e.closest("#learning-notes-list-radio")||d?.contains(e))&&ht&&d&&d.style.display!=="none")return t.preventDefault(),t.stopImmediatePropagation(),!1},I=t=>{if(!z||!d)return;const e=t.target;d.contains(e)||(t.preventDefault(),t.stopImmediatePropagation())},d.addEventListener("click",Q),document.addEventListener("pointerdown",I,{capture:!0}),document.addEventListener("mousedown",I,{capture:!0}),document.addEventListener("touchstart",I,{capture:!0}),document.addEventListener("click",tt,{capture:!0}))}function H(){Q&&d&&(d.removeEventListener("click",Q),Q=null),tt&&(document.removeEventListener("click",tt,{capture:!0}),tt=null),I&&(document.removeEventListener("pointerdown",I,{capture:!0}),document.removeEventListener("mousedown",I,{capture:!0}),document.removeEventListener("touchstart",I,{capture:!0}),I=null)}function Qe(){document.querySelector(".sidebar-navigation-btn-show")?.click(),setTimeout(()=>{document.querySelector("input#learning-notes-list-radio")?.click()},300)}function tn(){if(!d)return;const t=d.querySelector(".popup-close");t&&(t.onclick=null,t.addEventListener("click",e=>{e.preventDefault(),e.stopPropagation(),e.stopImmediatePropagation(),G()},!0))}async function en(){if(!B)throw new Error("There is no note data to save. Please select text and generate explanation before saving.");if(!W)throw new Error("There is no highlighted text. Please select text again and try the Explain AI feature.");if(!d)throw new Error("There is no popup available. Please try again.");let t="";if(y){const e=y.getBuffer().trim();e.length>0&&e!==_.LOADING&&(t=e)}if(!t)throw new Error("There is no explanation content to save as a note");if(B.highlightStartOffset===B.highlightEndOffset)throw new Error("Invalid highlight range");try{const e=await We({highlightedText:W,noteBody:t,token:window._token,notesData:B});if(!e.success)throw new Error(e.message||"Failed to save the note");document.dispatchEvent(new CustomEvent("explain-ai:note-saved",{detail:e.data})),B=null,W=null,G()}catch(e){throw new Error(e instanceof Error?e.message:"An unexpected error occurred while saving the note")}}function nn(){const t=document.getSelection();if(!t||t.rangeCount===0)return!1;const e=t.getRangeAt(0);if(e.collapsed)return!1;const a=e.toString().trim();if(a.length===0)return!1;const s=document.querySelector("#article-content");if(!s)return!1;try{return W=a,B=re(e,s),!0}catch(u){return console.error("Error generating note data:",u),B=null,W=null,!1}}function Lt(){d&&(ht=!0,d.style.cssText=`
    display: block !important;
    position: fixed !important;
    top: 100px !important;
    left: 100px !important;
    z-index: 99999 !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
  `,d.classList.add("explain-ai-protected"),de())}function G(){d&&(ht=!1,z=!1,d.style.display="none",d.classList.remove("explain-ai-protected"),gt&&(gt=!1,window.posthog?.capture("explain_with_ai_clicked",{course_id:ot?.courseId,page:window.location.pathname})),q(),H())}function it(t,e=!1){if(!d)return;let a=d.querySelector(".popup-explanation");if(!a){a=document.createElement("div"),a.className="popup-explanation";const s=d.querySelector(".popup-content");s?.insertBefore(a,s.firstChild)}if(t===_.LOADING||t===_.GENERAL_ERROR||t===_.MAX_LENGTH_ERROR){y&&(y.reset(),y=null),a.textContent=t;return}y||(y=new Me(a)),e||y.reset(),y.addChunk(t),t!==_.LOADING&&t.trim()!==""&&rn()}function on(){y&&y.complete()}function rt(){if(!d)return;ge();const t=d.querySelector(".popup-fallback");t&&(t.style.display="block"),ce(),Lt()}function q(){if(!d)return;const t=d.querySelector(".popup-fallback");t&&(t.style.display="none")}async function ue(){if(!ot){console.error("No previous request to retry");return}if(S?.remaining===0){fe(),G(),H();return}try{q(),it(_.LOADING),await se(ot,it)?q():rt()}catch(t){console.error("Retry failed:",t),rt()}}function ge(){if(!d)return;const t=d.querySelector(".popup-footer");t&&(t.style.display="none")}function rn(){if(!d)return;const t=d.querySelector(".popup-footer");t&&(t.style.display="block")}function an(t){if(!d)return;const e=d.querySelector(".credit-label");e&&(t?e.textContent="Kuota harian tersisa":e.textContent="Kuota tersisa")}function sn(t,e){const a=`${t}/${e}`,s=t===e,u=`${S?.reset?.nextResetAt}`;if(d){const l=d.querySelector(".credit-count");l&&(l.textContent=a)}he(s,a,Ct(u))}function he(t,e,a){const s=document.querySelector(".learning-notes__eai-credit-count");s&&(s.textContent=e,s.classList.toggle("learning-notes__eai-credit-count--limit",t));const u=document.querySelector(".learning-notes__eai-credit-action");u&&(t?(nt?(u.textContent=`Anda bisa mencoba kembali ${a} WIB.`,u.classList.remove("learning-notes__eai-credit-action--upgrade")):(u.innerHTML='<a href="/subscriptions/purchases" class="learning-notes__eai-credit-action--upgrade">Upgrade ke langganan <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4.66699 11.3337L11.3337 4.66699M11.3337 4.66699H4.66699M11.3337 4.66699V11.3337" stroke="#3B82F6" stroke-width="1.33333" stroke-linecap="round" stroke-linejoin="round"/></svg></a>',u.classList.add("learning-notes__eai-credit-action--upgrade")),u.classList.remove("d-none")):u.classList.add("d-none"))}function me(){d&&it(_.LOADING)}function ln(){ht=!1,z=!1,gt=!1,B=null,W=null,ot=null,S=null,y&&(y.reset(),y=null),H(),cn(),q(),d=null,U=null,O=null,_t=null,St=null,P=null,F=null}function cn(){P&&P.removeEventListener("click",bt),F&&F.removeEventListener("click",xt)}function fe(){const t=`${S?.reset?.nextResetAt}`,e={isPremiumUser:nt,resetTime:nt?Ct(t):void 0};Ve.showLimitModal(e)}window.addEventListener("beforeunload",H);const dn=Object.freeze(Object.defineProperty({__proto__:null,activateExplainWithAI:mt,cleanup:ln,completeStreaming:on,hideFallbackActions:q,hidePopup:G,initializeLoadingState:me,retryExplainWithAI:ue,setupExplainWithAIFeature:le,showFallbackActions:rt,showPopup:Lt},Symbol.toStringTag,{value:"Module"}));let ee=!1;function un(t,e){const a=document.getElementById("explain-with-ai-fab"),s=document.getElementById("explanation-popup"),u=s?.querySelector(".popup-close");a&&(ee||(ee=!0,le(a,s)),a.addEventListener("click",async()=>{const l=window.getSelection();if(!l)return;const h=Fe(l);if(!h.highlightedText.length)return;t(l,()=>{s.style.display="none"});const f=He(h);Object.assign(s.style,{left:`${h.firstRect.left}px`,top:`${f}px`,display:"block"}),mt()?.use({highlightedText:h.highlightedText,courseId:window.courseId})}),u?.addEventListener("click",()=>{s.style.display="none",e()}))}const gn=(t="",e=18,a=22)=>`
    <svg
        class="${t}"
        width="${e}"
        height="${a}"
        viewBox="0 0 18 22"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
    >
        <path
            d="M11 1.26953V5.40007C11 5.96012 11 6.24015 11.109 6.45406C11.2049 6.64222 11.3578 6.7952 11.546 6.89108C11.7599 7.00007 12.0399 7.00007 12.6 7.00007H16.7305M13 12H5M13 16H5M7 8H5M11 1H5.8C4.11984 1 3.27976 1 2.63803 1.32698C2.07354 1.6146 1.6146 2.07354 1.32698 2.63803C1 3.27976 1 4.11984 1 5.8V16.2C1 17.8802 1 18.7202 1.32698 19.362C1.6146 19.9265 2.07354 20.3854 2.63803 20.673C3.27976 21 4.11984 21 5.8 21H12.2C13.8802 21 14.7202 21 15.362 20.673C15.9265 20.3854 16.3854 19.9265 16.673 19.362C17 18.7202 17 17.8802 17 16.2V7L11 1Z"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
        />
    </svg>
`,hn=(t="",e=18,a=22)=>`
   
   <svg   
   class="${t}"
        width="${e}"
        height="${a}"
        viewBox="0 0 18 22" fill="none" xmlns="http://www.w3.org/2000/svg">
        <g clip-path="url(#clip0_70_148)">
        <path d="M3.74984 18.3332V14.1665M3.74984 5.83317V1.6665M1.6665 3.74984H5.83317M1.6665 16.2498H5.83317M10.8332 2.49984L9.38802 6.25722C9.15301 6.86825 9.03551 7.17376 8.85278 7.43074C8.69083 7.6585 8.49184 7.8575 8.26408 8.01945C8.00709 8.20218 7.70158 8.31968 7.09055 8.55469L3.33317 9.99984L7.09056 11.445C7.70158 11.68 8.00709 11.7975 8.26408 11.9802C8.49184 12.1422 8.69083 12.3412 8.85278 12.5689C9.03551 12.8259 9.15301 13.1314 9.38802 13.7425L10.8332 17.4998L12.2783 13.7425C12.5133 13.1314 12.6308 12.8259 12.8136 12.5689C12.9755 12.3412 13.1745 12.1422 13.4023 11.9802C13.6592 11.7975 13.9648 11.68 14.5758 11.445L18.3332 9.99984L14.5758 8.55469C13.9648 8.31968 13.6592 8.20217 13.4023 8.01945C13.1745 7.8575 12.9755 7.6585 12.8136 7.43074C12.6308 7.17376 12.5133 6.86825 12.2783 6.25722L10.8332 2.49984Z" stroke="#F59E0B" stroke-width="1.66667" stroke-linecap="round" stroke-linejoin="round"/>
        </g>
        <defs>
        <clipPath id="clip0_70_148">
        <rect width="20" height="20" fill="white"/>
        </clipPath>
        </defs>
    </svg>
`,mn=(t="",e=24)=>`
    <svg
        class="${t}"
        width="${e}"
        height="${e}"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
    >
        <path
            d="M12 13C12.5523 13 13 12.5523 13 12C13 11.4477 12.5523 11 12 11C11.4477 11 11 11.4477 11 12C11 12.5523 11.4477 13 12 13Z"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
        />
        <path
            d="M12 6C12.5523 6 13 5.55228 13 5C13 4.44772 12.5523 4 12 4C11.4477 4 11 4.44772 11 5C11 5.55228 11.4477 6 12 6Z"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
        />
        <path
            d="M12 20C12.5523 20 13 19.5523 13 19C13 18.4477 12.5523 18 12 18C11.4477 18 11 18.4477 11 19C11 19.5523 11.4477 20 12 20Z"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
        />
    </svg>
`,fn=(t="",e=24)=>`
    <svg
        class="${t}"
        width="${e}"
        height="${e}"
        viewBox="0 0 24 24"
        xmlns="http://www.w3.org/2000/svg"
    >
        <path
            fill-rule="evenodd"
            clip-rule="evenodd"
            d="M17.7071 3.29289C17.3166 2.90237 16.6834 2.90237 16.2929 3.29289L3.29289 16.2929C3.10536 16.4804 3 16.7348 3 17V20C3 20.5523 3.44772 21 4 21H7C7.26522 21 7.51957 20.8946 7.70711 20.7071L20.7071 7.70711C21.0976 7.31658 21.0976 6.68342 20.7071 6.29289L17.7071 3.29289ZM5 19V17.4142L13 9.41421L14.5858 11L6.58579 19H5ZM16 9.58579L18.5858 7L17 5.41421L14.4142 8L16 9.58579Z"
            fill="currentColor"
        />
    </svg>
`,pn=(t="",e=24)=>`
    <svg
        class="${t}"
        width="${e}"
        height="${e}"
        viewBox="0 0 24 24"
        xmlns="http://www.w3.org/2000/svg"
    >
        <path
            d="M20 4C20 4.55228 19.5523 5 19 5H5C4.44771 5 4 4.55228 4 4C4 3.44772 4.44771 3 5 3H19C19.5523 3 20 3.44772 20 4Z"
            fill="currentColor"
        />
        <path
            d="M9 10C9.55228 10 10 10.4477 10 11V17C10 17.5523 9.55228 18 9 18C8.44772 18 8 17.5523 8 17V11C8 10.4477 8.44772 10 9 10Z"
            fill="currentColor"
        />
        <path
            d="M13 11C13 10.4477 12.5523 10 12 10C11.4477 10 11 10.4477 11 11V17C11 17.5523 11.4477 18 12 18C12.5523 18 13 17.5523 13 17V11Z"
            fill="currentColor"
        />
        <path
            d="M16 11C16 10.4477 15.5523 10 15 10C14.4477 10 14 10.4477 14 11V17C14 17.5523 14.4477 18 15 18C15.5523 18 16 17.5523 16 17V11Z"
            fill="currentColor"
        />
        <path
            fill-rule="evenodd"
            clip-rule="evenodd"
            d="M4 7C4 6.44772 4.44772 6 5 6H19C19.5523 6 20 6.44772 20 7V21C20 21.5523 19.5523 22 19 22H5C4.44772 22 4 21.5523 4 21V7ZM6 8V20H18V8H6Z"
            fill="currentColor"
        />
    </svg>
`,wn=(t="",e=16)=>`
    <svg
        class="${t}"
        width="${e}"
        height="${e}"
        viewBox="0 0 24 24"
        fill="currentColor"
        xmlns="http://www.w3.org/2000/svg"
    >
        <path
            d="M4.70711 16.7071C4.31658 17.0976 3.68342 17.0976 3.29289 16.7071C2.90237 16.3166 2.90237 15.6834 3.29289 15.2929L11.2929 7.29289C11.6834 6.90237 12.3166 6.90237 12.7071 7.29289L20.7071 15.2929C21.0976 15.6834 21.0976 16.3166 20.7071 16.7071C20.3166 17.0976 19.6834 17.0976 19.2929 16.7071L12 9.41421L4.70711 16.7071Z"
        />
    </svg>
`,En=(t="",e=16)=>`
    <svg
        class="${t}"
        width="${e}"
        height="${e}"
        viewBox="0 0 16 16"
        fill="currentColor"
        xmlns="http://www.w3.org/2000/svg"
    >
        <path
                d="M3.13807 4.86128C2.87772 4.60093 2.45561 4.60093 2.19526 4.86128C1.93491 5.12163 1.93491 5.54374 2.19526 5.80409L7.5286 11.1374C7.78895 11.3978 8.21106 11.3978 8.47141 11.1374L13.8047 5.80409C14.0651 5.54374 14.0651 5.12163 13.8047 4.86128C13.5444 4.60093 13.1223 4.60093 12.8619 4.86128L8 9.72321L3.13807 4.86128Z"
        />
    </svg>
`;function ne(t){return`
        <li class="learning-notes-list__list-item" data-note-id="${t.id}">
            <div class="learning-notes-list__list-item__header">
                <div class="learning-notes-list__list-item__title">
                    ${t.moduleCategoryName!==null?`${T.sanitize(t.moduleCategoryName)} &bullet;`:""}
                    ${T.sanitize(t.moduleTitle)}
                </div>
                <div>
                    <button type="button" class="learning-notes-list__list-item__more-tools-button" data-toggle="dropdown" aria-expanded="false">
                        ${mn("learning-notes-list__list-item__more-tools-icon",24)}
                    </button>
                    <div class="dropdown-menu dropdown-menu-left">
                        ${t.noteBody.startsWith(D)?"":`
                        <button
                            class="dropdown-item d-inline-flex align-items-center"
                            style="gap: 12px"
                            data-target="learning-notes-list-controller.edit"
                            data-note-id="${t.id}"
                        >
                            ${fn("learning-notes-list__edit-icon",24)}
                            Edit
                        </button>
                        `}
                        <form
                            method="POST"
                            action="${window._sharedData.generateDeleteExistingNoteEndpoint(t.id)}"
                            class="dropdown-item d-inline-flex align-items-center"
                            data-target='learning-notes-list-controller.removeForm'
                            data-note-id="${t.id}"
                        >
                            <input type="hidden" name="_method" value="DELETE" >
                            <input type="hidden" name="_token" value="${window._token}" >
                            <button
                                class="dropdown-item p-0 d-inline-flex align-items-center"
                                style="gap: 12px"
                                data-target="learning-notes-list-controller.remove"
                                data-note-id="${t.id}"
                            >
                                ${pn("learning-notes-list__remove-icon",24)}
                                Hapus
                            </button>
                        </form>
                    </div>
                </div>
            </div>
            <div class="learning-notes-list__list-item__highlighted-text">
                <div>${gn("learning-notes-list__list-item__icon")}</div>
                <div class="learning-notes-list__list-item__highlighted-text__content">
                    <div
                        class="learning-notes-list__list-item__highlighted-text__text"
                        data-target="learning-notes-list-controller.highlightedText"
                        data-note-id="${t.id}"
                    >
                        ${t.moduleId===window._sharedData.tutorialId?`
                                    <button
                                        class="learning-notes-list__list-item__highlighted-text__text-button border-0 bg-transparent text-left"
                                        type="button"
                                        data-show-in-the-lesson-content-button
                                        data-note-id="${t.id}"
                                    >
                                        ${T.sanitize(t.highlightedText)}
                                    </button>
                                `:`
                                    <a
                                        class="learning-notes-list__list-item__highlighted-text__text-link text-left"
                                        href="${window._sharedData.generateTutorialShowUrlForNoteHighlight(t.id)}"
                                    >
                                        ${T.sanitize(t.highlightedText)}
                                    </a>
                                `}
                    </div>
                    <div>
                        <button
                            class="learning-notes-list__list-item__highlighted-text__expand-button d-none"
                            data-target="learning-notes-list-controller.expandHighlightedText"
                            data-note-id="${t.id}"
                        >
                            Lihat lebih banyak
                            ${wn("",10)}
                        </button>
                        <button
                            class="learning-notes-list__list-item__highlighted-text__collapse-button d-none"
                            data-target="learning-notes-list-controller.collapseHighlightedText"
                            data-note-id="${t.id}"
                        >
                            Lihat lebih sedikit
                            ${En("",10)}
                        </button>
                    </div>
                </div>
            </div>
            <div class="learning-notes-list__list-item__note-body">
                <div class="learning-notes-list__list-item__title">Catatan</div>
                <div>
                    ${t.noteBody.length<1?`
                                <button
                                    class="learning-notes-list__list-item__add-note-button"
                                    data-target="learning-notes-list-controller.updateEmptyNoteButton"
                                    data-note-id="${t.id}"
                                >
                                    &plus; Tambah Catatan
                                </button>
                            `:t.noteBody.startsWith(D)?ie(t.noteBody.substring(D.length).trimStart()):T.sanitize(t.noteBody)}
                </div>
            </div>
            <div class="learning-notes-list__list-item__more-info">
                <div class="learning-notes-list__list-item__title">
                    ${new Date(t.createdAt).toLocaleDateString("id-ID",{day:"numeric",month:"long",year:"numeric"})}
                </div>
            </div>
        </li>
    `}const $={VISIBILITY_TOGGLE:"note-indicator-visibility-toggle",CUSTOM_CONTEXTMENU_TOGGLE_KEY:"is-custom-contextmenu-nonactive"},yn={CUSTOM_CONTEXTMENU_TOGGLE_EVENT:"toggle:classroom:custom-contextmenu"};function oe(t,e){if(e.startsWith(D)){const a=document.createElement("span");a.innerHTML=hn("mr-2 learning-note-indicators__item-icon"),t.firstElementChild?.querySelector("svg")?.replaceWith(a.firstElementChild)}return t}J(window).on("load",()=>{const t=document.querySelector("#tutorial-content"),e=document.querySelector("#article-content"),a=document.querySelector("#user-selection-floating-buttons-backdrop"),s=a.querySelector("#user-selection-floating-buttons-popover"),u=s.querySelector("#copy-text-floating-button"),l=s.querySelector("#show-learning-note-creation-form-floating-button"),h=document.querySelector("#learning-note-creation-backdrop"),f=document.querySelector("#learning-note-creation-popover"),k=document.querySelector("#learning-note-creation-form"),v=J("#note-body-textarea"),j=document.querySelector('input[name="highlighted_text"]'),A=document.querySelector('input[name="highlight_start_offset"]'),R=document.querySelector('input[name="highlight_end_offset"]'),w=document.querySelector('input[name="highlight_prefix"]'),N=document.querySelector('input[name="highlight_suffix"]'),b=document.querySelector("#learning-note-update-backdrop"),x=document.querySelector("#learning-note-update-popover"),at=x.querySelectorAll('[name="highlight_color"]'),Z=J("#note-body-textarea-update"),X=document.querySelector("#learning-note-indicators"),pe=document.querySelector("#learning-note-indicators-dropdown-toggle-button-template"),ft=document.querySelector("#note-indicator-visibility-toggle"),we=document.querySelectorAll("button[data-show-in-the-lesson-content-button]"),Ee=document.querySelector("#learning-note-onboarding-start-button");let V=null,pt=[];function kt(){return Number(localStorage.getItem($.CUSTOM_CONTEXTMENU_TOGGLE_KEY)??0)===0}const Nt={isCustomContextmenuActive:kt()};window.addEventListener(yn.CUSTOM_CONTEXTMENU_TOGGLE_EVENT,()=>{Nt.isCustomContextmenuActive=kt()});function Tt(i){const n=i.cloneContents();i.deleteContents();const o=document.createElement("mark");return o.append(n.cloneNode(!0)),i.insertNode(o),{oldContent:n,markElement:o}}function It(i,n){i.deleteContents(),i.insertNode(n),i.commonAncestorContainer.normalize()}function ye({whenAppear:i,whenDisappear:n}){i(),a.classList.remove("d-none"),s.removeAttribute("aria-hidden"),s.focus(),a.onclick=o=>{const r=o.target,c=!Array.from(s.querySelectorAll("button,a")).some(m=>m.contains(r));s.contains(r)&&c||(a.classList.add("d-none"),a.onclick=null,s.setAttribute("aria-hidden","true"))}}function At({whenAppear:i,whenDisappear:n,displayNote:o=!0}){i(),h.classList.remove("d-none"),o?(f.classList.remove("d-none"),f.removeAttribute("aria-hidden"),f.focus()):(f.classList.add("d-none"),f.setAttribute("aria-hidden","true")),h.onclick=r=>{const c=r.target;f.contains(c)||(n(),h.classList.add("d-none"),h.onclick=null,f.setAttribute("aria-hidden","true"))}}function Mt(i){j.value=i?.highlightedText??"",A.value=i?.highlightStartOffset.toString()??"0",R.value=i?.highlightEndOffset.toString()??"0",w.value=i?.highlightPrefix??"",N.value=i?.highlightSuffix??"",i===null&&(v.val(""),v.froalaEditor("html.set",""))}function Bt({whenAppear:i,whenDisappear:n}){i(),b.classList.remove("d-none"),x.removeAttribute("aria-hidden"),x.focus(),b.onclick=o=>{const r=o.target;x.contains(r)||(n(),b.classList.add("d-none"),b.onclick=null,x.setAttribute("aria-hidden","true"))}}function st(i="",n=null){at.forEach(o=>{o.value===n&&(o.checked=!0)}),Z.val(T.sanitize(i)),Z.froalaEditor("html.set",T.sanitize(i))}function ve(){v.froalaEditor("destroy");const i=v.data("placeholder")??"Masukkan catatan kamu di sini";v.froalaEditor({placeholderText:T.sanitize(i),toolbarButtons:["bold","italic","underline","|","formatOL","formatUL"]})}function Ot(){Z.froalaEditor("destroy");const i=Z.data("placeholder")??"Masukkan catatan kamu di sini";Z.froalaEditor({placeholderText:T.sanitize(i),toolbarButtons:["bold","italic","underline","|","formatOL","formatUL"]})}function qt(i){function n(g){return g.parentElement===null?null:g.parentElement===e?g:n(g.parentElement)}function o(){const g=i.startContainer.parentElement.parentElement.tagName.toLowerCase()==="span"&&i.startContainer.parentElement.parentElement.classList.contains("learning-note-indicator-text")&&i.startContainer.parentElement.parentElement.hasAttribute("data-note-id"),C=i.endContainer.parentElement.parentElement.tagName.toLowerCase()==="span"&&i.endContainer.parentElement.parentElement.classList.contains("learning-note-indicator-text")&&i.endContainer.parentElement.parentElement.hasAttribute("data-note-id");if(g||C)return!0;const M=document.createTreeWalker(e,NodeFilter.SHOW_ALL);let lt=null,ct=null,wt=[];for(;M.nextNode()&&!ct;)M.currentNode===i.startContainer&&(lt=M.currentNode),lt&&(wt=wt.concat(M.currentNode)),M.currentNode===i.endContainer&&(ct=M.currentNode);return!!wt.find(dt=>dt instanceof HTMLElement?dt.tagName.toLowerCase()==="span"&&dt.classList.contains("learning-note-indicator-text")&&dt.hasAttribute("data-note-id"):!1)}if(o())throw new Error("Note taking not allowed!");const r=n(i.startContainer.parentElement),c=n(i.endContainer.parentElement);if(!r||!c)throw new Error("Error was happened, please try again!");if(["panel-code","coderunner-widget"].some(g=>r.classList.contains(g)))throw new Error("Note taking not allowed!");const p=["img","iframe","table","pre","textarea"];if(p.includes(r.tagName.toLowerCase()))throw new Error("Note taking not allowed!");if(p.some(g=>r.querySelector(g)))throw new Error("Note taking not allowed!");if(r!==c)throw new Error("Note taking not allowed!");function E(g){return g.tagName.toLowerCase()==="li"?g:g.parentElement===null||g.parentElement===e?null:E(g.parentElement)}const L=E(i.startContainer.parentElement);if(L){const g=E(i.endContainer.parentElement);if(L!==g)throw new Error("Note taking not allowed!")}}function be(i,n){s.style.left=`${i}px`,s.style.top=`${n}px`}function xe(i,n){f.style.left=`${i}px`,f.style.top=`${n}px`}function Rt(i,n){x.style.left=`${i}px`,x.style.top=`${n}px`}document.addEventListener("mouseup",i=>{if(!Nt.isCustomContextmenuActive)return;const n=document.getSelection();!n||!n.toString().length||e?.contains(n.anchorNode)&&(mt(),ye({whenAppear(){be(i.clientX,i.clientY);try{qt(n.getRangeAt(0)),l.disabled=!1,l.style.pointerEvents="auto",l.parentElement?.setAttribute("tabindex","-1"),l.parentElement?.setAttribute("title","Buat catatan")}catch{l.disabled=!0,l.style.pointerEvents="none",l.parentElement?.setAttribute("tabindex","0"),l.parentElement?.setAttribute("title","Fitur catatan belum mendukung konten ini")}},whenDisappear(){}}))}),l.addEventListener("click",()=>{const i=document.getSelection();if(!i)return;const n=i.getRangeAt(0);try{qt(n);const o=re(n,e),{oldContent:r,markElement:c}=Tt(n),m=new ResizeObserver(()=>{const p=c.getBoundingClientRect();xe(Math.min(p.left+p.width/2-10,window.innerWidth-(f.offsetWidth+100)),Math.min(p.top+p.height-10,window.innerHeight-(f.offsetHeight+100)))});At({whenAppear(){i.empty(),m.observe(t.parentElement),ve(),Mt(o)},whenDisappear(){It(n,r),Mt(null),m.disconnect()}})}catch(o){console.error(o)}}),u.addEventListener("click",async()=>{const i=window.getSelection(),n=i?.toString();if(n)try{if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(n);return}const o=document.createElement("textarea");o.value=n,o.setAttribute("readonly",""),o.style.position="fixed",o.style.top="-1000px",document.body.appendChild(o),o.select();const r=document.execCommand("copy");if(document.body.removeChild(o),!r)throw new Error("Failed to copy text")}catch(o){console.error(o)}finally{i?.empty()}});function $t(i){function n(E){return E.parentElement===null?null:E.parentElement===e&&E instanceof HTMLElement?E:n(E.parentElement)}h.dispatchEvent(new Event("click"));const o=ut([i.data]),[r]=Ft(o);if(r.status===et.ORPHANED)return;const c=Kt(e,r);Dt(c,r);const m=n(c.startContainer);if(m){const E={parent:m,notes:[r]};Zt([E]);const L=Array.from(m.querySelectorAll(".learning-note-indicator-text[data-note-id]")).filter(C=>Number(C.dataset.noteId)!==r.id).map(C=>Number(C.dataset.noteId));L.sort();const g=document.querySelector(`.learning-note-indicators__item[data-note-indicators-id="${JSON.stringify(L)}"]`);if(g===null)Pt([E]),Ht([E]),Gt(E);else{const C=g.querySelector(".dropdown-toggle"),M=Number(C?.dataset.notesLength)+(r.noteBody.length>0?1:0);C?.setAttribute("data-notes-length",M.toString());const lt=[...L,r.id];g.setAttribute("data-note-indicators-id",JSON.stringify(lt));const ct=Qt(r,Y);g.querySelector(".dropdown-menu")?.appendChild(oe(ct,r.noteBody))}}const p=document.querySelector("#notes-list");if(p)p.innerHTML+=ne(r);else{const E=document.querySelector("#sidebar-nav-content-body");E&&(document.querySelector("#sidebar-nav-content-header")?.classList.remove("d-none"),E.innerHTML=`
                    <ul id="notes-list" class="learning-notes-list__list">
                        ${ne(r)}
                    </ul>
                `)}}k?.addEventListener("submit",async i=>{i.preventDefault();const n=new FormData(i.currentTarget,i.submitter);try{const o=await ae({_token:n.get("_token"),note_body:n.get("note_body"),highlighted_text:n.get("highlighted_text"),highlight_start_offset:n.get("highlight_start_offset"),highlight_end_offset:n.get("highlight_end_offset"),highlight_prefix:n.get("highlight_prefix"),highlight_suffix:n.get("highlight_suffix"),highlight_color:n.get("highlight_color")});$t(o)}catch(o){if(console.log(o),o instanceof Error){alert(o.message);return}}finally{i.submitter?.removeAttribute("disabled")}}),document.addEventListener("explain-ai:note-saved",i=>{$t(i.detail)}),un((i,n)=>{const o=i.getRangeAt(0),{oldContent:r}=Tt(o);At({whenAppear(){h.classList.add("d-none"),i.empty()},whenDisappear(){n(),It(o,r)},displayNote:!1})});function Dt(i,n){const o=document.createElement("span");o.classList.add("learning-note-indicator-text"),o.style.setProperty("--highlight-color",n.highlightColorHex),o.setAttribute("role","button"),o.setAttribute("tabindex","0"),o.setAttribute("onkeydown","if(event.key === 'Enter' || event.key === ' '){ event.preventDefault();this.click(); }"),o.setAttribute("data-note-id",n.id.toString()),n.noteBody.startsWith(D)&&o.setAttribute("data-is-ai-note","true"),o.addEventListener("click",()=>{if(n.noteBody.startsWith(D)){Y(n.id);return}const r=new ResizeObserver(()=>{const c=o.getBoundingClientRect();Rt(Math.min(c.left-10,window.innerWidth-(x.offsetWidth+100)),Math.min(c.top+c.height-10,window.innerHeight-(x.offsetHeight+100)))});Bt({whenAppear(){r.observe(t.parentElement),Ot(),st(n.noteBody,n.highlightColor);const c=x.querySelector("form"),m=c.action.split("/");m[Jt(m,"learning-notes")+1]=n.id.toString(),c.action=m.join("/")},whenDisappear(){st(),r.disconnect()}})}),o.appendChild(i.extractContents()),i.insertNode(o)}function Ce(i){function n(r){return r.parentElement===null?null:r.parentElement===e?r:n(r.parentElement)}return i.reduce((r,c)=>{if(c.status==="orphaned")return r;const m=document.querySelector(`.learning-note-indicator-text[data-note-id="${c.id}"]`),p=n(m.parentElement);if(!p)return r;const E=r.find(({parent:L})=>L===p);return E?(E.notes.push(c),r):r.concat({parent:p,notes:[c]})},[])}function Pt(i){i.forEach(({parent:n,notes:o})=>{if(o.every(g=>g.noteBody.length<1))return;const r=document.createElement("div");r.classList.add("dropdown-menu"),o.forEach(g=>{const C=Qt(g,Y);r.appendChild(oe(C,g.noteBody))});const m=Array.from(n.querySelectorAll(".learning-note-indicator-text[data-note-id]")).map(g=>Number(g.dataset.noteId));m.sort();let p=document.querySelector(`.learning-note-indicators__item[data-note-indicators-id="${JSON.stringify(m)}"]`);if(p===null){const g=o.map(({id:C})=>C);g.sort(),p=document.createElement("div"),p.classList.add("learning-note-indicators__item","dropright"),p.dataset.noteIndicatorsId=JSON.stringify(g)}const E=pe.content.cloneNode(!0);let L=o.filter(g=>g.noteBody.length>0).length;E.firstElementChild?.setAttribute("data-notes-length",L.toString()),p.append(E,r),X.appendChild(p)}),document.body.addEventListener("click",n=>{const o=n.target,r=Array.from(document.querySelectorAll(".learning-notes-list__list-item.learning-notes-list__list-item--state-active"));r.some(m=>!m.contains(o))&&r.forEach(m=>{m.classList.remove("learning-notes-list__list-item--state-active")})})}function Ft(i){return i.map(o=>$e(o,e))}function _e(i){i.map(o=>{if(o.status===et.ORPHANED)return{note:o,range:null};const r=Kt(e,o);return{note:o,range:r}}).forEach(({note:o,range:r})=>{r&&Dt(r,o)})}function Ht(i){i.flatMap(n=>n.notes).forEach(n=>{const o=document.querySelector(`.learning-note-indicators__item__button[data-note-id="${n.id}"]`),r=document.querySelector(`.learning-note-indicator-text[data-note-id="${n.id}"]`);o?.addEventListener("mouseenter",()=>{r?.classList.add("learning-note-indicator-text--state-active")}),o?.addEventListener("mouseleave",()=>{r?.classList.remove("learning-note-indicator-text--state-active")})})}function Gt({parent:i,notes:n}){const o=n.map(({id:c})=>c);o.sort();let r=document.querySelector(`[data-note-indicators-id="${JSON.stringify(o)}"]`);r&&(r.style.top=`${i.offsetTop}px`,r.style.left=`calc(${i.offsetLeft}px - 40px)`)}function Se(){const i=new ResizeObserver(o=>{o.forEach(()=>{pt.forEach(r=>{Gt(r)})})}),n=document.getElementById("classroom-grid-chatbot")!==null;i.observe(n?t:t.parentElement)}async function Le(){const i=qe("Siap Belajar","Siap belajar dengan Learning Note");V=Oe("learning-note-onboarding");const n=new Et.Step(V,{id:"highlight",cancelIcon:{enabled:!0,label:"Lewati onboarding"},buttons:[Xt],when:{show(){vt(this.getElement())}},text:yt({title:"Tandai Bagian Penting,<br>Belajar Jadi Lebih Fokus",description:"Tandai bagian yang penting biar gampang dicari lagi nanti. Hemat waktu, fokus ke inti pelajaran.",videoLink:"https://assets.cdn.dicoding.com/videos/intro-highlight-text-feature.mp4"})}),o=new Et.Step(V,{id:"note-taking",cancelIcon:{enabled:!0,label:"Lewati onboarding"},buttons:[Yt,Xt],when:{show(){vt(this.getElement())}},text:yt({title:"Catat Sekarang,<br>Pahami Lebih Dalam",description:"Menulis catatan membantu mengingat dan memahami pelajaran lebih cepat. Semua insight tersimpan aman di sini.",videoLink:"https://assets.cdn.dicoding.com/videos/intro-note-taking-feature.mp4"})}),r=new Et.Step(V,{id:"note-access",cancelIcon:{enabled:!0,label:"Lewati onboarding"},buttons:[Yt,i],when:{show(){vt(this.getElement())}},text:yt({title:"Akses Cepat ke<br> Catatanmu",description:"Akses semua catatan kapan saja. Ulangi materi dan segarkan ingatan dengan cepat.",videoLink:"https://assets.cdn.dicoding.com/videos/intro-note-access-feature.mp4"})});V.addSteps([n,o,r]),Ee?.addEventListener("click",async()=>{await V?.start()})}function Vt(i){if(i){X.classList.remove("d-none"),X.removeAttribute("aria-hidden"),document.querySelectorAll(".learning-note-indicator-text").forEach(n=>{n.classList.remove("learning-note-indicator-text--state-hide"),n.setAttribute("tabindex","0"),n.removeAttribute("aria-hidden")});return}X.classList.add("d-none"),X.setAttribute("aria-hidden","true"),document.querySelectorAll(".learning-note-indicator-text").forEach(n=>{n.classList.add("learning-note-indicator-text--state-hide"),n.setAttribute("tabindex","-1"),n.setAttribute("aria-hidden","true")})}function Ut(i){const n=document.querySelector("#note-indicator-visibility-eye-on-template"),o=document.querySelector("#note-indicator-visibility-eye-off-template");ft.replaceChild(i?n.content.cloneNode(!0):o.content.cloneNode(!0),ft.querySelector("svg"))}function ke(){localStorage[$.VISIBILITY_TOGGLE]||localStorage.setItem($.VISIBILITY_TOGGLE,"true");const i=localStorage.getItem($.VISIBILITY_TOGGLE)==="true";Vt(i),Ut(i),ft.onclick=()=>{const n=localStorage.getItem($.VISIBILITY_TOGGLE)==="true";n?localStorage.setItem($.VISIBILITY_TOGGLE,"false"):localStorage.setItem($.VISIBILITY_TOGGLE,"true"),Vt(!n),Ut(!n)}}function Ne(){let i=-1;we.forEach(o=>{o.addEventListener("click",r=>{const c=r.target,m=Number(c.dataset.noteId);m<1||isNaN(m)||(i=m,jt(),zt(m),Y(m))})});const n=new URLSearchParams(window.location.search);if(n.has("highlight-note")){const o=Number(n.get("highlight-note"));if(o<1||isNaN(o))return;i=o,jt(),zt(o),Y(o)}e?.addEventListener("click",o=>{const r=o.target;if(i>0){const c=document.querySelector(`.learning-note-indicator-text[data-note-id="${i}"]`);c&&!c.contains(r)&&c.classList.remove("learning-note-indicator-text--state-active")}})}function Wt(i,n){const o=document.querySelector(`.learning-notes-list__list-item[data-note-id="${i}"]`);setTimeout(()=>{o?.classList.add("learning-notes-list__list-item--state-active"),o?.scrollIntoView({behavior:"smooth",block:"start"})},n)}function Y(i){const n=document.getElementById("classroom-grid-chatbot");if(n){const c=n.classList.contains("collapsed");c&&document.querySelector('[data-sidebar-chatbot-target="collapseBtn"]')?.click(),document.querySelector("input#learning-notes-list-radio")?.parentElement?.click(),Wt(i,c?300:0);return}const o=document.querySelector(".sidebar-navigation-btn-show"),r=o?.classList.contains("show")??!1;r&&o?.click(),o?.nextElementSibling&&(document.querySelector("input#learning-notes-list-radio")?.parentElement?.click(),Wt(i,r?500:0))}function zt(i){const n=document.querySelector(`.learning-note-indicator-text[data-note-id="${i}"]`);if(!n){const o=document.querySelector(`.learning-notes-list__list-item[data-note-id="${i}"]`);o?!o.querySelector(".learning-notes-list__list-item__highlighted-text__text-link")&&J("#note-was-obsolete-modal").modal("show"):J("#note-not-found-modal").modal("show");return}n.classList.add("learning-note-indicator-text--state-active"),n.scrollIntoView({behavior:"smooth",block:"center"})}function jt(){document.querySelectorAll(".learning-note-indicator-text").forEach(n=>{n.classList.remove("learning-note-indicator-text--state-active")})}function Te(){function i(o){const r=document.querySelector(`.learning-note-indicator-text[data-note-id="${o}"]`);r&&r.dataset.isAiNote!=="true"&&Bt({whenAppear(){r.scrollIntoView({behavior:"auto",block:"center"});const c=r.getBoundingClientRect();Rt(c.left,c.top+10),Ot(),st();const m=x.querySelector("form"),p=m.action.split("/");p[Jt(p,"learning-notes")+1]=o.toString(),m.action=p.join("/")},whenDisappear(){st()}})}const n=new URLSearchParams(window.location.search);if(n.has("highlight-note")&&n.has("note-edit")){const o=Number(n.get("highlight-note"));if(o<1||isNaN(o)||!(n.get("note-edit")==="true"))return;i(o)}}function Zt(i){const n=new Map;for(const o of pt)n.set(o.parent,o.notes.slice());for(const o of i){const r=n.get(o.parent);r?n.set(o.parent,r.concat(o.notes)):n.set(o.parent,o.notes.slice())}pt=Array.from(n,([o,r])=>({parent:o,notes:r}))}function Ie(){e?.querySelectorAll("ul li p, ol li p").forEach(i=>{const n=i.closest("li");if(n){for(;i.firstChild;)n.insertBefore(i.firstChild,i);i.remove()}})}setTimeout(()=>{for(const r of document.querySelectorAll(".js-render-markdown"))r.innerHTML=ie((r.textContent??"").trim()),r.closest(".learning-notes-list__list-item")?.querySelector('[data-target="learning-notes-list-controller.edit"]')?.remove();Le();const i=ut(window.learningNotes);Ie();const n=Ft(i);_e(n);const o=Ce(n);Zt(o),Pt(o),Ht(o),Se(),ke(),Te(),Ne()})});
