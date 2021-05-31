% Use this editor as a MiniZinc scratch book
include "globals.mzn";
include "chuffed.mzn";

int :nr; % Number of Rounds
int :wordsize=64; % word size
int :bit=320; % Number of bits

int: R = nr-1;% just because starting R from 1 doesnt work!
array [0..R+1,0..4, 0..wordsize-1] of var 0..1: statestart; % before permutation
array [0..R,0..4, 0..wordsize-1] of var 0..1: stateend; % after permutation
var int: obj; % objective value
constraint obj <= 320;
constraint obj >= 1;

% Forward Permutation
constraint forall (r in 0..R-1) (
let {
    array [0..wordsize-1] of var 0..1:tmp=RRot(array1d(0..wordsize-1,[stateend[r,0,j] | j in 0..wordsize-1]),19),
    array [0..wordsize-1] of var 0..1:tmpb=RRot(array1d(0..wordsize-1,[stateend[r,0,j] | j in 0..wordsize-1]),28),
    array [0..wordsize-1] of var 0..1:tmpc=array1d(0..wordsize-1,[stateend[r,0,j] | j in 0..wordsize-1]),
    array [0..wordsize-1] of var 0..1:tmpd=array1d(0..wordsize-1,[statestart[r+1,0,j] | j in 0..wordsize-1])
}
in 
  Xor4(tmp, tmpb, tmpc, tmpd)
);

constraint forall (r in 0..R-1) (
let {
    array [0..wordsize-1] of var 0..1:tmp=RRot(array1d(0..wordsize-1,[stateend[r,1,j] | j in 0..wordsize-1]),61),
    array [0..wordsize-1] of var 0..1:tmpb=RRot(array1d(0..wordsize-1,[stateend[r,1,j] | j in 0..wordsize-1]),39),
    array [0..wordsize-1] of var 0..1:tmpc=array1d(0..wordsize-1,[stateend[r,1,j] | j in 0..wordsize-1]),
    array [0..wordsize-1] of var 0..1:tmpd=array1d(0..wordsize-1,[statestart[r+1,1,j] | j in 0..wordsize-1])
}
in 
  Xor4(tmp, tmpb, tmpc, tmpd)
);

constraint forall (r in 0..R-1) (
let {
    array [0..wordsize-1] of var 0..1:tmp=RRot(array1d(0..wordsize-1,[stateend[r,2,j] | j in 0..wordsize-1]),1),
    array [0..wordsize-1] of var 0..1:tmpb=RRot(array1d(0..wordsize-1,[stateend[r,2,j] | j in 0..wordsize-1]),6),
    array [0..wordsize-1] of var 0..1:tmpc=array1d(0..wordsize-1,[stateend[r,2,j] | j in 0..wordsize-1]),
    array [0..wordsize-1] of var 0..1:tmpd=array1d(0..wordsize-1,[statestart[r+1,2,j] | j in 0..wordsize-1])
}
in 
 % tmpd=Xor2(Xor2(tmp,tmpb),tmpc)
  Xor4(tmp, tmpb, tmpc, tmpd)

);

constraint forall (r in 0..R-1) (
let {
    array [0..wordsize-1] of var 0..1:tmp=RRot(array1d(0..wordsize-1,[stateend[r,3,j] | j in 0..wordsize-1]),10),
    array [0..wordsize-1] of var 0..1:tmpb=RRot(array1d(0..wordsize-1,[stateend[r,3,j] | j in 0..wordsize-1]),17),
    array [0..wordsize-1] of var 0..1:tmpc=array1d(0..wordsize-1,[stateend[r,3,j] | j in 0..wordsize-1]),
    array [0..wordsize-1] of var 0..1:tmpd=array1d(0..wordsize-1,[statestart[r+1,3,j] | j in 0..wordsize-1])
}
in 
  %tmpd=Xor2(Xor2(tmp,tmpb),tmpc)
  Xor4(tmp, tmpb, tmpc, tmpd)

);

constraint forall (r in 0..R-1) (
let {
    array [0..wordsize-1] of var 0..1:tmp=RRot(array1d(0..wordsize-1,[stateend[r,4,j] | j in 0..wordsize-1]),7),
    array [0..wordsize-1] of var 0..1:tmpb=RRot(array1d(0..wordsize-1,[stateend[r,4,j] | j in 0..wordsize-1]),41),
    array [0..wordsize-1] of var 0..1:tmpc=array1d(0..wordsize-1,[stateend[r,4,j] | j in 0..wordsize-1]),
    array [0..wordsize-1] of var 0..1:tmpd=array1d(0..wordsize-1,[statestart[r+1,4,j] | j in 0..wordsize-1])
}
in 
 % tmpd=Xor2(Xor2(tmp,tmpb),tmpc)
  Xor4(tmp, tmpb, tmpc, tmpd)
);

predicate checkActive(array[int] of var 0..1: a) = 
            let { int: len = length(a) }
          in
            sum(i in 1..len) (a[i]) > 0;

% a function takes in array and val, return right rotate by val (in binary)
% Right rotation of X by val positions
function array[0..wordsize-1] of var 0..1: RRot(array[0..(wordsize-1)] of var 0..1: X, int: val)=
array1d(0..wordsize-1, [X[(wordsize+j-val) mod wordsize] | j in 0..wordsize-1]);

%%%%Obj
constraint obj = sum(row(sboxes,0));

output [show([Sboxes[i] | i in 0..1]) ++ "\n"]
 ++ [show(sboxes[0,i]) | i in 0..63] ++ ["\n"]
  ++ [show(sboxes[1,i]) | i in 0..63] ++ ["\n"];



%solve
ann: search_ann;
search_ann = int_search(array1d(sboxes), random_order, indomain_min, complete);

array [0..1,0..63] of var 0..1: sboxes;
constraint forall(j in 0..63) (sboxes[0,j] = checkActive([stateend[0,i,j] | i in 0..4]));
constraint forall(j in 0..63) (sboxes[1,j] = checkActive([statestart[1,i,j] | i in 0..4]));

% for printing purposes
array[0..1] of var int: Sboxes;
constraint forall (k in 0..1)(
  Sboxes[k] = sum(j in 0..63)(sboxes[k,j])
);

constraint sum(j in 0..63)(sboxes[1,j]) = sbox;
int: sbox;

solve :: restart_constant(1000)
:: search_ann satisfy;




    
% use restart_constant(10000) if there is an element of randomness

% eliminating symmetries
constraint checkActive([statestart[1,i,63] | i in 0..4]) = 1; 


% XOR of 4 arrays equals zero
predicate Xor4(
  array[0..wordsize-1] of var 0..1: a, 
  array[0..wordsize-1] of var 0..1: b,
  array[0..wordsize-1] of var 0..1: c, 
  array[0..wordsize-1] of var 0..1: d)=
    forall(i in 0..wordsize-1)(a[i]+b[i]+c[i]+d[i] in {0,2,4});

nr = 2;







