from pathlib import Path
import json
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1]
(root/'evidence').mkdir(exist_ok=True)
out=root/'ultra/source/console/branding';out.mkdir(exist_ok=True,parents=True)
# Editable geometric reconstruction of the project's existing vLLM mark.
left=[(76,164),(228,164),(228,448)]
right=[(310,155),(454,80),(354,448),(228,448)]
# A small, separate bottom-left accent leaves the original vLLM V unobscured.
bolt=[(109,329),(69,405),(98,405),(82,480),(159,384),(124,384),(147,329)]
bolt_opacity=.78
def svg(background=False,mono=False):
 bg='<rect x="16" y="16" width="480" height="480" rx="108" fill="#101c30"/>' if background else ''
 shapes=''.join('<polygon points="'+' '.join(f'{x},{y}' for x,y in pts)+'" fill="'+color+'"/>' for pts,color in [(left,'#ffb510' if not mono else 'currentColor'),(right,'#2999f5' if not mono else 'currentColor')])
 shapes+='<polygon points="'+' '.join(f'{x},{y}' for x,y in bolt)+'" fill="'+('#ffe05a' if not mono else 'currentColor')+'" stroke="#101c30" stroke-width="8" stroke-linejoin="round" opacity="'+str(bolt_opacity)+'"/>'
 return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><title>VLLM-SM75 Accelerated</title>'+bg+shapes+'</svg>'
(out/'favicon.svg').write_text(svg(True),encoding='utf-8')
(out/'mark-transparent.svg').write_text(svg(),encoding='utf-8')
(out/'mark-mono.svg').write_text(svg(mono=True),encoding='utf-8')
def raster(size,background=True,mask=False):
 factor=4;im=Image.new('RGBA',(size*factor,size*factor));d=ImageDraw.Draw(im);scale=size*factor/512
 if background:d.rounded_rectangle((16*scale,16*scale,496*scale,496*scale),108*scale,fill='#101c30')
 def pts(points):
  if mask:points=[(256+(x-256)*.8,256+(y-256)*.8) for x,y in points]
  return [(x*scale,y*scale) for x,y in points]
 d.polygon(pts(left),fill='#ffb510');d.polygon(pts(right),fill='#2999f5')
 accent=Image.new('RGBA',im.size);a=ImageDraw.Draw(accent)
 a.line(pts(bolt+[bolt[0]]),fill='#101c30',width=max(1,round(8*scale)),joint='curve');a.polygon(pts(bolt),fill='#ffe05a')
 accent.putalpha(accent.getchannel('A').point(lambda value:round(value*bolt_opacity)))
 im=Image.alpha_composite(im,accent)
 return im.resize((size,size),Image.Resampling.LANCZOS)
for n in [32,64,180,192,512]:raster(n).save(out/f'icon-{n}.png')
raster(256).save(out/'favicon.ico',sizes=[(16,16),(32,32),(48,48)])
# Maskable has full opaque canvas, including outside the icon safe zone.
m=Image.new('RGBA',(512,512),'#101c30');m.alpha_composite(raster(512,True,True));m.save(out/'maskable-512.png')
(out/'manifest.webmanifest').write_text(json.dumps({'name':'VLLM-SM75 工作台','short_name':'SM75','start_url':'/','display':'standalone','background_color':'#101c30','theme_color':'#101c30','icons':[{'src':'/brand/icon-192.png','sizes':'192x192','type':'image/png'},{'src':'/brand/icon-512.png','sizes':'512x512','type':'image/png'},{'src':'/brand/maskable-512.png','sizes':'512x512','type':'image/png','purpose':'maskable'}]},ensure_ascii=False,indent=2),encoding='utf-8')
preview=Image.new('RGB',(1000,440),'#eaf0f7')
preview.paste(raster(300,False),(80,30),raster(300,False));preview.paste(raster(300),(570,30),raster(300))
d=ImageDraw.Draw(preview);d.text((100,360),'A / transparent geometric mark',fill='#101c30');d.text((585,360),'B / app tile (selected)',fill='#101c30')
for i,n in enumerate([16,32,48,64]):im=raster(n);preview.paste(im,(600+i*85,390-n//2),im)
preview.save(root/'evidence/brand-preview.png')
