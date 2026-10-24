/** JSON-driven editable PowerPoint builder. Use the supplied Codex artifact runtime. */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';

const args = process.argv.slice(2);
if (args.length !== 2) throw new Error('Usage: node scripts/build_presentation.mjs content.json NEW_output_directory');
const workspaceDir = process.cwd();
const contentPath = path.resolve(args[0]);
const outputDir = path.resolve(args[1]);
const relativeOutput = path.relative(workspaceDir, outputDir);
if (relativeOutput.startsWith('..') || path.isAbsolute(relativeOutput)) throw new Error('Output must be inside the workspace');
await fs.access(outputDir).then(() => { throw new Error('Output directory exists. Use a new revision directory.'); }, () => {});
const content = JSON.parse(await fs.readFile(contentPath, 'utf8'));
if (!Array.isArray(content.slides) || !content.slides.length) throw new Error('slides must be a nonempty array');
const runtimeRoot = process.env.CODEX_PRESENTATION_RUNTIME ?? path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime');
const modules = process.env.RUNTIME_NODE_MODULES ?? path.join(runtimeRoot, 'dependencies/node/node_modules');
process.env.RUNTIME_NODE_MODULES = modules;
const python = process.env.RUNTIME_PYTHON ?? path.join(runtimeRoot, 'dependencies/python/python.exe');
const skillDir = process.env.PRESENTATION_SKILL_DIR ?? path.join(os.homedir(), '.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations');
const requireRuntime = createRequire(path.join(modules, '__presentation__.cjs'));
const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(requireRuntime.resolve('@oai/artifact-tool')).href);
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const font = 'Arial';
const durations = content.slides.map(s => Number(s.duration ?? 50));
if (durations.some(v => !Number.isFinite(v) || v <= 0)) throw new Error('Each duration must be positive seconds');
if (content.duration != null && Math.abs(durations.reduce((a,b)=>a+b,0)-content.duration)>0.001) throw new Error('Slide durations must sum to content.duration');
for (const s of content.slides) {
  if (!s.title || !s.notes) throw new Error('Every slide requires title and English speaker notes');
  if ((s.claims ?? []).length > 4) throw new Error('At most four claims per slide. Split dense content.');
  if ([s.table,s.chart,s.image].filter(Boolean).length>1) throw new Error('table, chart and image are mutually exclusive');
  if (s.chart && (s.chart.series ?? []).some(v => v.values.length !== s.chart.categories.length || v.values.some(n=>!Number.isFinite(n)))) throw new Error('Chart values must be finite and match categories');
  if (s.table && (s.table.length > 8 || s.table.some(row=>row.length !== s.table[0].length))) throw new Error('Table must be rectangular and at most eight rows');
}
const stagingDir = path.join(path.dirname(outputDir), '.' + path.basename(outputDir) + '-build');
await fs.mkdir(stagingDir, {recursive:true});
await fs.mkdir(path.join(outputDir, 'slides'), {recursive:true});
const presentation = Presentation.create({slideSize:{width:1280,height:720}});
presentation.theme.defaultFont = font;
function text(slide, value, left, top, width, height, size, bold=false, color='#142235') {
  const shape = slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
  shape.text = value;
  shape.text.style = {typeface:font,fontSize:size,bold,color};
  return shape;
}
const tables=[], charts=[], sourceImages=[];
for (const [i,s] of content.slides.entries()) {
  const slide=presentation.slides.add();
  slide.background.fill='#FFFFFF';
  text(slide,s.title,64,48,1152,108,i===0&&s.layout==='cover'?56:44,true);
  const claims=s.claims ?? [];
  const evidence=Boolean(s.table||s.chart||s.image);
  const claimWidth=evidence&&claims.length?355:1152;
  const claimTop=i===0&&s.layout==='cover'?260:184;
  claims.forEach((claim,j)=>text(slide,claim,64,claimTop+j*(evidence?105:92),claimWidth,evidence?100:84,28));
  const left=claims.length?460:76, width=claims.length?744:1120;
  if (s.table) {
    tables.push(i+1);
    const values=s.table.map(row=>row.map(String));
    const table=slide.tables.add({rows:values.length,columns:values[0].length,left,top:190,width,height:Math.min(410,values.length*62),values});
    table.borders.assign({fill:'#CBD2DA',width:0.8,style:'solid'});
    table.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({textStyle:{typeface:font,fontSize:24,color:'#142235'},margins:{left:10,right:10,top:10,bottom:10}});
    table.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:'#EAF0F5',textStyle:{typeface:font,fontSize:24,bold:true,color:'#142235'}});
  }
  if(s.chart) {
    charts.push(i+1);
    const c=s.chart;
    const chart=slide.charts.add(c.type??'bar',{position:{left,top:172,width,height:420},categories:c.categories,series:c.series.map((x,j)=>({...x,fill:['#24557A','#6B929E','#A7BAC1'][j%3]})),hasLegend:c.series.length>1,legend:{position:'bottom',textStyle:{fontSize:22}},barOptions:{direction:c.direction??'column',grouping:'clustered'},xAxis:{textStyle:{fontSize:22},title:{text:c.xLabel??'',textStyle:{fontSize:22}}},yAxis:{minimumScale:c.minimum??0,textStyle:{fontSize:22},numberFormatCode:c.numberFormat??'0.0',title:{text:c.yLabel??'',textStyle:{fontSize:22}},majorGridlines:{fill:'#D8DEE5',width:0.6}},dataLabels:{showValue:c.showValues??false,textStyle:{fontSize:22},position:'outEnd'}});
    applyPresentationChartFont(chart,{fontFamily:font});
  }
  if(s.image) {
    const imagePath=path.resolve(workspaceDir,typeof s.image==='string'?s.image:s.image.path);
    const bytes=await fs.readFile(imagePath);
    const hash=createHash('sha256').update(bytes).digest('hex');
    const extension=path.extname(imagePath).toLowerCase();
    const contentType={'.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg'}[extension];
    if(!contentType) throw new Error('Scientific figure images must be PNG or JPEG');
    const image=slide.images.add({blob:bytes,contentType,alt:s.image.alt??s.title,fit:'contain',position:{left,top:172,width,height:430}});
    image.lockAspectRatio=true;
    sourceImages.push({slide:i+1,path:path.relative(workspaceDir,imagePath).replaceAll('\\','/'),sha256:hash});
  }
  if(s.caption) text(slide,s.caption,64,618,1140,50,23,false,'#42556B');
  text(slide,String(i+1).padStart(2,'0'),1170,667,44,28,18,false,'#607085');
  const figure=sourceImages.find(x=>x.slide===i+1);
  slide.speakerNotes.text=`${s.notes}\n\nSources:\n${(s.citations??[]).join('\n')}${figure?`\nFigure: ${figure.path}\nFigure SHA256: ${figure.sha256}`:''}`;
}
const candidatePath=path.join(stagingDir,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const finalPath=path.join(outputDir,'airline_delay.pptx');
await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:python,integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:content.slides.length,requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:charts,materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(stagingDir,'validation.json')});
// Render the validated final file, rather than only the in-memory candidate.
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for (const [i,slide] of finalDeck.slides.items.entries()) {
  const image=await finalDeck.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(outputDir,'slides',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await image.arrayBuffer()));
}
function stamp(seconds) {return new Date(Math.round(seconds*1000)).toISOString().slice(11,23).replace('.',',');}
function captionChunks(notes) {
  const chunks=[];
  for(const sentence of notes.match(/[^.!?]+[.!?]+|[^.!?]+$/g)??[notes]) {
    let current='';
    for(const word of sentence.trim().split(/\s+/)) {
      if(current && `${current} ${word}`.length>78) {chunks.push(current);current=word;}
      else current=current?`${current} ${word}`:word;
    }
    if(current) chunks.push(current);
  }
  return chunks;
}
function captionLines(chunk) {
  const words=chunk.split(' '); let lines=[''];
  for(const word of words) {const i=lines.length-1;if(lines[i] && `${lines[i]} ${word}`.length>42)lines.push(word);else lines[i]=lines[i]?`${lines[i]} ${word}`:word;}
  return lines.join('\n');
}
let time=0, cue=1, srt=[], script=[`# ${content.title??'Airline delay'}\n\nEnglish speaker script for human narration.\n`];
for (const [i,s] of content.slides.entries()) {
  script.push(`## Slide ${i+1}: ${s.title}\n\n${time.toFixed(1)}–${(time+durations[i]).toFixed(1)} seconds\n\n${s.notes}\n\nSources: ${(s.citations??[]).join('; ')}\n`);
  const chunks=captionChunks(s.notes);
  const weights=chunks.map(x=>x.trim().split(/\s+/).length), total=weights.reduce((a,b)=>a+b,0);
  let cursor=time;
  for (const [j,chunk] of chunks.entries()) { const end=j===chunks.length-1?time+durations[i]:cursor+durations[i]*weights[j]/total; srt.push(`${cue++}\n${stamp(cursor)} --> ${stamp(end)}\n${captionLines(chunk)}\n`);cursor=end; }
  time+=durations[i];
}
await fs.writeFile(path.join(outputDir,'speaker_script_en.md'),script.join('\n'));
await fs.writeFile(path.join(outputDir,'captions_en.srt'),srt.join('\n'));
await fs.copyFile(contentPath,path.join(outputDir,'content.json'));
await fs.writeFile(path.join(outputDir,'source_images.json'),JSON.stringify(sourceImages,null,2));
await fs.writeFile(path.join(outputDir,'timing.json'),JSON.stringify({duration:time,narration:'not-recorded',captionedDemo:true,slides:content.slides.map((s,i)=>({slide:i+1,png:`slides/slide-${String(i+1).padStart(2,'0')}.png`,duration:durations[i]}))},null,2));
console.log(JSON.stringify({pptx:finalPath,slides:content.slides.length,duration:time,receipt:path.join(stagingDir,'validation.json'),previews:path.join(outputDir,'slides'),narration:'No audio generated. Script and timed captions ready for human recording.'},null,2));
