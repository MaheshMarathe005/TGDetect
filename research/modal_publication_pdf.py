"""Compile the editable IEEE manuscript on CPU only; no GPU reservation."""
from pathlib import Path
import modal
root=Path(__file__).resolve().parent.parent/'research paper/publication'
image=modal.Image.debian_slim(python_version='3.10').apt_install(
    'texlive-latex-base','texlive-latex-recommended','texlive-fonts-recommended','texlive-publishers')
app=modal.App('tg-detect-paper-cpu-build')


@app.function(image=image,cpu=2,memory=2048,timeout=600)
def compile_pdf(files):
    import subprocess
    directory=Path('/tmp/paper');directory.mkdir(exist_ok=True)
    for name,content in files.items():
        path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    logs=[]
    for cmd in (['pdflatex','-interaction=nonstopmode','-halt-on-error','paper.tex'],
                ['bibtex','paper'],['pdflatex','-interaction=nonstopmode','-halt-on-error','paper.tex'],
                ['pdflatex','-interaction=nonstopmode','-halt-on-error','paper.tex']):
        p=subprocess.run(cmd,cwd=directory,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        logs.append(p.stdout)
        if p.returncode:raise RuntimeError(p.stdout[-12000:])
    return {'pdf':(directory/'paper.pdf').read_bytes(),'log':'\n'.join(logs),'bbl':(directory/'paper.bbl').read_bytes()}


@app.local_entrypoint()
def main():
    files={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*')
           if p.is_file() and (p.suffix in ('.tex','.bib') or p.parent.name=='figures' and p.suffix=='.pdf')}
    result=compile_pdf.remote(files)
    (root/'paper.pdf').write_bytes(result['pdf']);(root/'build.log').write_text(result['log'])
    (root/'paper.bbl').write_bytes(result['bbl']);print('Saved',root/'paper.pdf')
