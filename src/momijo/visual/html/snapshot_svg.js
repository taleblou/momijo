// tools/snapshot_svg.js
const puppeteer = require('puppeteer');
const fs = require('fs');
(async () => {
  const [infile, outfile] = process.argv.slice(2);
  if (!infile || !outfile) { console.error('Usage: node snapshot_svg.js in.html out.svg'); process.exit(1); }
  const browser = await puppeteer.launch({headless:true,args:['--no-sandbox']});
  const page = await browser.newPage();
  await page.goto('file://' + require('path').resolve(infile));
  await page.waitForTimeout(1200);
  const svgs = await page.$$eval('svg', els => els.map(e => e.outerHTML));
  fs.writeFileSync(outfile, svgs.join('\n'));
  await browser.close();
})();