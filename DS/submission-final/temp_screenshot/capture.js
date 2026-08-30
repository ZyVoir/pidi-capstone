const puppeteer = require('puppeteer');

(async () => {
  console.log('Launching browser...');
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();
  
  // Set window size
  await page.setViewport({ width: 1600, height: 1200 });
  
  const url = 'http://localhost:3000/public/dashboard/ad0434e2-85df-4ed8-8a53-a01a41fa1efd';
  console.log(`Navigating to ${url}...`);
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 60000 });
  
  console.log('Waiting 20 seconds for charts to render...');
  await new Promise(resolve => setTimeout(resolve, 20000));
  
  const destPath = '/Users/zyvoir/Documents/PIDI/Capstone/DS/submission-final/submission/William_dicoding-dashboard.png';
  console.log(`Taking screenshot to ${destPath}...`);
  await page.screenshot({ path: destPath, fullPage: false });
  
  console.log('Done!');
  await browser.close();
})();
