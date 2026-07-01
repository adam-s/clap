import pkg from '/Users/adamsohn/Projects/separate/node_modules/playwright-core/index.js';
const { chromium } = pkg;

const url = process.argv[2] || 'http://localhost:4180/';
const out = process.argv[3] || '/tmp/clap-shot.png';
const stage = parseInt(process.argv[4] || '0', 10);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 1600 }, deviceScaleFactor: 2 });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(String(e)));
await page.goto(url, { waitUntil: 'networkidle' });
await page.waitForTimeout(800);
// advance the stage playhead by clicking next N times
for (let i = 0; i < stage; i++) {
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(250);
}
await page.waitForTimeout(900);
const vp = process.argv[5] === 'vp';
await page.screenshot({ path: out, fullPage: !vp });
await browser.close();
console.log('shot ->', out, 'stage', stage);
if (errors.length) { console.log('CONSOLE ERRORS:\n' + errors.join('\n')); process.exit(2); }
