// tools/snapshot.js
const puppeteer = require('puppeteer');
(async () => {
  const [infile, outfile] = process.argv.slice(2);
  if (!infile || !outfile) { console.error('Usage: node snapshot.js in.html out.png'); process.exit(1); }
  const browser = await puppeteer.launch({headless:true,args:['--no-sandbox']});
  const page = await browser.newPage();
  await page.goto('file://' + require('path').resolve(infile));
  await page.waitForTimeout(1200);
  await page.screenshot({path: outfile, fullPage: true});
  await browser.close();
})();