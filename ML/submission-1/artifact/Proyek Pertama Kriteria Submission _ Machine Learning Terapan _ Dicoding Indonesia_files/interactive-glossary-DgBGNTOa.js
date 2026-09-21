import r from"jquery";import{S as s}from"./shepherd-Bpa11e5q.js";import{t as c}from"./interactive-glossary-usage-tracker-Cnmavl70.js";const d={ONBOARDING:"interactive-glossary-onboarding"};document.addEventListener("DOMContentLoaded",function(){let t=null;const l=window.glossary.map(e=>({id:e.id,term:e.term,definition:e.definition}));function i(e){return l.find(n=>n.term.toLowerCase()===e.toLowerCase())??{id:0,term:e,definition:"Definisi dari istilah ini tidak ditemukan. Mohon untuk menghubungi tim support kami untuk bantuan lebih lanjut."}}document.querySelectorAll(".interactive-glossary").forEach(e=>{const a=e.textContent?.trim()||"",n=i(a);r(e)?.tooltip({trigger:"manual",placement:"top",title:n.definition,customClass:"interactive-glossary-tooltip"}),u(e,n)});async function o(){t=new s.Tour({tourName:"onboarding",useModalOverlay:!0,defaultStepOptions:{classes:"interactive-glossary-tour",scrollTo:!1},confirmCancel:!1,exitOnEsc:!0,keyboardNavigation:!0});const e={text:"Lanjut belajar",classes:"shepherd-button__go",label:"Lanjut belajar dengan Interactive Glossary",disabled:!1,action(){this.complete()}},a=new s.Step(t,{id:"glossary-access",cancelIcon:{enabled:!0,label:"Lewati onboarding"},buttons:[e],when:{show(){}},text:()=>`
          <div class="interactive-glossary-onboarding__content">
              <div class="interactive-glossary-onboarding__header">
                  <video class="interactive-glossary-onboarding__video" autoplay muted loop playsinline>
                      <source src="https://assets.cdn.dicoding.com/videos/intro-interactive-glossary.webm" type="video/webm">
                      <source src="https://assets.cdn.dicoding.com/videos/intro-interactive-glossary.mp4" type="video/mp4">
                  </video>
              </div>
              <div class="interactive-glossary-onboarding__body">
                  <div class="text-center mb-4">
                      <span class="interactive-glossary-onboarding__badge">
                          <span class="interactive-glossary-onboarding__badge--new-text">Baru!</span>
                          <span>Interactive Glossary</span>
                      </span>
                  </div>
                  <h3 class="interactive-glossary-onboarding__title">
                      Cari Makna dan Definisi Istilah Kini Lebih Mudah
                  </h3>
                  <p class="interactive-glossary-onboarding__description">
                      Mari coba Interactive Glossary saat belajar.
                      <br>
                      Arahkan kursor atau klik pada istilah yang digarisbawahi untuk mengetahui makna atau definisinya.
                  </p>
              </div>
          </div>
      `});s.on("complete",()=>{localStorage.setItem(d.ONBOARDING,"true")}),s.on("cancel",()=>{localStorage.setItem(d.ONBOARDING,"true")}),t.addSteps([a])}o()});function u(t,l){let i,o=0;t.addEventListener("mouseenter",()=>{o=performance.now(),r(t).tooltip("show");const e=t.getAttribute("aria-describedby");if(e===null)throw new Error("Tooltip element does not exists in DOM!");const a=document.querySelector(`[id="${e}"]`);if(a===null)throw new Error("Tooltip element does not exists in DOM!");setTimeout(()=>{a.addEventListener("mouseenter",()=>{clearTimeout(i)}),a.addEventListener("mouseleave",()=>{i=setTimeout(()=>{r(t).tooltip("hide")},100)})},0)}),t.addEventListener("mouseleave",async()=>{i=setTimeout(()=>{r(t).tooltip("hide")},100);const e=parseFloat(((performance.now()-o)/1e3).toFixed(1));e>=0&&(await c(l,e),o=0)})}
