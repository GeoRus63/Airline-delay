/** Verify the production caption functions without importing or building a PPTX. */
import fs from 'node:fs/promises';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const [contentArg,outputArg,oldSrtArg]=process.argv.slice(2);
if(!contentArg || !outputArg) throw new Error('Usage: node scripts/validate_presentation_captions.mjs content.json NEW_output_directory [old.srt]');
const output=path.resolve(outputArg);
await fs.access(output).then(()=>{throw new Error('Output directory exists');},()=>{});
const builder=path.resolve('scripts/build_presentation.mjs');
const code=await fs.readFile(builder,'utf8');
const start=code.indexOf('function stamp(seconds)'), end=code.indexOf('let time=0, cue=1',start);
assert(start>=0 && end>start,'Production caption function boundaries not found');
// Evaluate only the exact production functions; no artifact tool, files or PPTX execution.
const {stamp,captionChunks,captionLines}=vm.runInNewContext(`${code.slice(start,end)}\n({stamp,captionChunks,captionLines})`);
const words=text=>text.trim().split(/\s+/);
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
function checkText(notes) {
  const chunks=Array.from(captionChunks(notes));
  const lines=chunks.map(captionLines);
  assert.deepEqual(words(lines.join(' ')),words(notes),'Caption words or punctuation changed');
  for(const text of lines) {
    assert(text.split('\n').length<=2);
    assert(text.split('\n').every(line=>line.length<=42));
  }
  return {chunks,lines};
}
const edgeCases=[
  'The MAE is 13.655 minutes, the CI is [-2.403, 3.012], and the fraction is 8.8068%.',
  'This preserves e.g. model identifiers, 0.495, 1.0 and U.S. punctuation without sentence splitting.',
  'These words describe the support set and the monthly differences across the airports in the analysis. End here.',
  Array(17).fill('word').join(' '),
  `${'a'.repeat(41)} ${'b'.repeat(41)} ${'c'.repeat(41)} ${'d'.repeat(41)}`,
  'A short script.',
];
edgeCases.forEach(checkText);
assert(Array.from(captionChunks(edgeCases[3])).every(chunk=>words(chunk).length>=4),'Short trailing group was not balanced');
assert.throws(()=>captionChunks('x'.repeat(43)),/exceeds 42/);
const contentPath=path.resolve(contentArg), contentBytes=await fs.readFile(contentPath);
const content=JSON.parse(contentBytes);
let clock=0,number=1,all=[],slides=[];
for(const [index,slide] of content.slides.entries()) {
  const duration=Number(slide.duration??50), {chunks,lines}=checkText(slide.notes);
  const counts=chunks.map(chunk=>words(chunk).length), total=counts.reduce((a,b)=>a+b,0);
  const start=clock; let cursor=start,cues=[];
  for(let i=0;i<chunks.length;i++) {
    const end=i===chunks.length-1?start+duration:cursor+duration*counts[i]/total;
    const startMs=Math.round(cursor*1000),endMs=Math.round(end*1000);
    assert(endMs>startMs);
    if(all.length) assert.equal(startMs,all.at(-1).end_ms,'Rounded SRT timeline has a gap or overlap');
    const cue={number:number++,slide:index+1,start_ms:startMs,end_ms:endMs,duration_seconds:(endMs-startMs)/1000,word_count:counts[i],text:lines[i]};
    cues.push(cue);all.push(cue);cursor=end;
  }
  assert.equal(cues[0].start_ms,Math.round(start*1000));
  assert.equal(cues.at(-1).end_ms,Math.round((start+duration)*1000));
  slides.push({slide:index+1,duration_seconds:duration,notes_sha256:sha(slide.notes),word_count:total,cue_count:cues.length,minimum_cue_seconds:Math.min(...cues.map(c=>c.duration_seconds)),minimum_cue_words:Math.min(...counts),word_sequence_identical:true,maximum_line_characters:Math.max(...lines.flatMap(text=>text.split('\n').map(line=>line.length)))});
  clock+=duration;
}
assert.equal(clock,content.duration??600);
assert.equal(all[0].start_ms,0);assert.equal(all.at(-1).end_ms,Math.round(clock*1000));
assert(slides.every(s=>s.minimum_cue_words>=4),'Actual deck has an isolated cue of fewer than four words');
const srt=all.map(cue=>`${cue.number}\n${stamp(cue.start_ms/1000)} --> ${stamp(cue.end_ms/1000)}\n${cue.text}\n`).join('\n');
function parseOld(srt) {
  const seconds=value=>{const [h,m,s,ms]=value.split(/[:,]/).map(Number);return h*3600+m*60+s+ms/1000;};
  return srt.replaceAll('\r\n','\n').trim().split(/\n\s*\n/).map(block=>{const [number,times,...text]=block.split('\n');const [start,end]=times.split(' --> ').map(seconds);return {number:Number(number),start,end,duration_seconds:Number((end-start).toFixed(3)),text:text.join('\n')};});
}
let before=null;
if(oldSrtArg) {
  const oldPath=path.resolve(oldSrtArg),bytes=await fs.readFile(oldPath),cues=parseOld(bytes.toString('utf8'));
  before={path:oldPath,sha256:sha(bytes),cue_count:cues.length,minimum_cue_seconds:Math.min(...cues.map(c=>c.duration_seconds)),first_four_cues:cues.slice(0,4),short_cues_below_one_second:cues.filter(c=>c.duration_seconds<1)};
}
await fs.mkdir(output,{recursive:true});
await fs.writeFile(path.join(output,'captions_after.srt'),srt);
await fs.writeFile(path.join(output,'builder_validated.mjs'),code);
const receipt={created_utc:new Date().toISOString(),status:'passed caption-only validation; no PPTX or final video built',content:{path:contentPath,sha256:sha(contentBytes)},builder:{path:builder,sha256:sha(code),tested_function_source:code.slice(start,end)},validator:{path:path.resolve(process.argv[1]),sha256:sha(await fs.readFile(process.argv[1]))},before,after:{path:path.join(output,'captions_after.srt'),sha256:sha(srt),duration_seconds:clock,slides:slides.length,cue_count:all.length,minimum_cue_seconds:Math.min(...all.map(c=>c.duration_seconds)),minimum_cue_words:Math.min(...all.map(c=>c.word_count)),maximum_lines:Math.max(...all.map(c=>c.text.split('\n').length)),maximum_line_characters:Math.max(...all.flatMap(c=>c.text.split('\n').map(line=>line.length))),first_four_cues:all.slice(0,4),every_note_word_and_punctuation_preserved:true,rounded_timeline_contiguous:true,original_slide_durations_preserved:true},edge_cases_passed:edgeCases.length,oversize_word_rejection_passed:true,slides,limitations:'Timing remains proportional to word count, not synchronized to recorded speech. Full rendered-caption readability and final 600-second video QA remain required.'};
await fs.writeFile(path.join(output,'validation.json'),JSON.stringify(receipt,null,2));
console.log(JSON.stringify({status:receipt.status,...receipt.after,validation:path.join(output,'validation.json')},null,2));
