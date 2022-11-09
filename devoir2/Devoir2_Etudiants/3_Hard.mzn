include "globals.mzn";

int: n = 5;

array[1..n] of int: t_b = [2,2,3,2,1]; % top to bottom
array[1..n] of int: b_t = [3,2,1,3,4]; % bottom to top
array[1..n] of int: l_r = [4,1,3,2,3]; % left to right
array[1..n] of int: r_l = [1,4,2,2,2]; % right to left

array[1..n, 1..n] of var 1..n: x;      % hauteur des tours à chaque position

% Contrainte sur l'uniciter d'hauteur des tours de chaque segment horizontal.
constraint forall(k in 1..n)(all_different(x[1..n,k]));
% Contrainte sur l'uniciter d'hauteur des tours de chaque segment vertical.
constraint forall(k in 1..n)(all_different(x[k,1..n]));
% Contrainte sur l'observateur de gauche à droite 
constraint forall(i in 1..n)( l_r[i] == sum([1 | j in 2..n where forall(k in 1..j-1)(x[i,j]>x[i,k])]) + 1);
% Contrainte sur l'observateur de droite à gauche 
constraint forall(i in 1..n)( r_l[i] == sum([1 | j in 2..n where forall(k in 1..j-1)(x[i,(n+1)-j]>x[i,(n+1)-k])]) + 1);
% Contrainte sur l'observateur de haut en bas 
constraint forall(j in 1..n)( t_b[j] == sum([1 | i in 2..n where forall(k in 1..i-1)(x[i,j]>x[k,j])]) + 1);
% Contrainte sur l'observateur de bas en haut 
constraint forall(j in 1..n)( b_t[j] == sum([1 | i in 2..n where forall(k in 1..i-1)(x[(n+1)-i,j]>x[(n+1)-k,j])]) + 1);

output 

["Configuration : \n"]

++

[ show(x[i,j]) ++ if j = n then "\n" else "," endif | i in 1..n, j in 1..n]
