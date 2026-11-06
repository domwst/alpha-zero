import assert from 'node:assert/strict';
import test from 'node:test';
import { terminalPosition, unpackPosition } from '../src/replayPosition.ts';
const pack=(cells:number[])=>Array.from({length:91},(_,i)=>cells.slice(i*4,i*4+4).reduce((n,v,j)=>n|(v<<(j*2)),0));
test('final move is applied and canonical colors swap for old archives',()=>{
 const cells=Array(361).fill(0);cells[0]=1;cells[1]=2;
 const final=terminalPosition({plies:[{state:{state:pack(cells)},action:{x:0,y:2}}]})!;
 assert.deepEqual(final.slice(0,4),[2,1,2,0]);
 assert.equal(final.filter(Boolean).length,3);
});
test('terminal state is preferred, including zero-ply archives',()=>{
 const cells=Array(361).fill(0);cells[5]=2;
 assert.deepEqual(terminalPosition({plies:[],terminal_state:{state:pack(cells)}}),cells);
 assert.deepEqual(unpackPosition(pack(cells)),cells);
});
test('missing, occupied, out-of-range or malformed actions are not invented',()=>{
 const state={state:pack(Array(361).fill(1))};
 for(const action of [null,{x:19,y:0},{x:0,y:0},{x:0.5,y:1}])assert.equal(terminalPosition({plies:[{state,action}]}),null);
 assert.equal(terminalPosition({plies:[]}),null);
});
