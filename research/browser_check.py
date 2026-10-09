"""Browser acceptance of the copied interface against real Modal artifacts."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent
out=ROOT/'artifacts/teammate_frontend_review'
out.mkdir(parents=True,exist_ok=True)
errors=[];checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/home/mahesh/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:3000',wait_until='domcontentloaded')
    page.locator('select[aria-label="Global active model"] option').nth(8).wait_for(state='attached',timeout=60000)
    page.get_by_text('153,056',exact=False).first.wait_for()
    page.screenshot(path=str(out/'overview.png'),full_page=True)
    for tab in ['events','graph','chains','datasets','artifacts','model','analytics']:
        page.locator(f'#nav-{tab}').click()
        if tab=='graph':
            canvas=page.get_by_test_id('temporal-graph');canvas.wait_for(timeout=60000);page.wait_for_timeout(1200)
            assert int(canvas.get_attribute('data-edge-count'))>0
            pixels=canvas.evaluate('(c)=>Array.from(c.getContext("2d").getImageData(0,0,c.width,c.height).data).filter((v,i)=>i%4===3&&v>0).length')
            assert pixels>5000;checks.append('real graph edges and painted pixels')
        elif tab=='events':
            page.locator('tbody tr').first.wait_for(timeout=60000)
            assert page.get_by_text('Ground truth',exact=True).count()>0
        else:page.wait_for_timeout(1800)
        page.screenshot(path=str(out/f'{tab}.png'),full_page=True);checks.append(f'{tab} navigation')
    page.locator('#nav-model').click()
    page.get_by_role('button',name='Matched comparison',exact=True).click()
    page.get_by_text('Matched comparison · 9 saved runs',exact=True).wait_for(timeout=60000)
    assert '0.9326' in page.locator('body').inner_text()
    page.get_by_label('Comparison threshold').select_option('test_default_0_5')
    assert '0.9522' in page.locator('body').inner_text()
    page.screenshot(path=str(out/'comparison.png'),full_page=True);checks.append('both saved comparison thresholds')
    page.get_by_label('Active model',exact=True).select_option('matched:TGNN_Q_A_C:1')
    page.wait_for_timeout(1200)
    assert page.get_by_label('Global active model').input_value()=='matched:TGNN_Q_A_C:1'
    page.wait_for_timeout(21000)
    assert page.get_by_label('Global active model').input_value()=='matched:TGNN_Q_A_C:1'
    checks.append('model selection survives health polling')
    page.get_by_role('button',name='Live inference',exact=True).click()
    page.get_by_role('button',name='Load checkpoint and process 64 events',exact=True).click()
    page.get_by_text('Position: 64',exact=False).wait_for(timeout=60000)
    assert page.locator('tbody tr').count()==64
    page.screenshot(path=str(out/'live_inference.png'),full_page=True);checks.append('64 events scored by actual checkpoint')
    page.get_by_role('button',name='Evaluation',exact=True).click()
    page.locator('tbody tr').first.wait_for(timeout=60000);checks.append('selected model evaluation')
    page.locator('#nav-graph').click();page.get_by_test_id('temporal-graph').wait_for(timeout=60000)
    page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(1000)
    page.screenshot(path=str(out/'mobile.png'),full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),'Mobile viewport overflow'
    checks.append('mobile viewport');browser.close()
assert not errors,errors
(out/'browser_verification.json').write_text(json.dumps(dict(status='passed',checks=checks,page_errors=errors),indent=2))
print('Browser acceptance passed:',', '.join(checks))
