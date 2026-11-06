import assert from 'node:assert/strict';
import test from 'node:test';
import {api,streamAnalysis,heartbeatStamp} from '../src/serviceApi.ts';
test('proxy HTML and broken JSON errors preserve HTTP status, without leaking markup',async()=>{
 const original=globalThis.fetch;
 try{
  for(const [type,body] of [['text/html','<html>bad gateway</html>'],['application/json','{']]){
   globalThis.fetch=async()=>new Response(body,{status:502,headers:{'Content-Type':type}});
   await assert.rejects(api('analysis'),/Request failed \(HTTP 502\)/);
   await assert.rejects(streamAnalysis({}, {authenticated:true},()=>{},new AbortController().signal),/Search failed \(HTTP 502\)/);
  }
  globalThis.fetch=async()=>Response.json({error:'Worker is busy'},{status:409});
  await assert.rejects(api('analysis'),/Worker is busy/);
 }finally{globalThis.fetch=original;}
});
test('heartbeat timestamp uses server time even if the browser clock differs',()=>{
 const now=Date.now()/1000,server=now-3600;
 assert.equal(heartbeatStamp(server-3,server),new Date((server-3)*1000).toLocaleTimeString(undefined,{hourCycle:'h23'}));
 assert.equal(heartbeatStamp(server-700,server),new Date((server-700)*1000).toLocaleString(undefined,{hourCycle:'h23'}));
});
