include "globals.mzn";

var int: nombre;
array[1..4] of var 1..9: chiffres;

% Constraint c’est un nombre pair.
constraint (chiffres[4] mod 2 = 0);
% Constraint le chiffre 0 n’est pas présent dans le nombre.
constraint forall ( i in 1..4)(chiffres[i]!=0);
% Constraint les 4 chiffres sont différents.
constraint all_different(chiffres);
% Constraint le chiffre à la position des milliers est supérieur à celui à la position des centaines.
% Constraint le chiffre à la position des dizaines est inférieur à celui à la position des unités.
% Constraint le chiffre à la position des centaines est supérieur à celui à la position des unités.
% Constraint la somme des 4 chiffres est supérieure à 15.
constraint chiffres[1]>chiffres[2] /\ chiffres[3]<chiffres[4] /\ chiffres[2]>chiffres[4] /\ sum(chiffres) > 15 ;
% Constraint le produit des 3 derniers chiffres (chiffre à la position des centaines × chiffre à la position des dizaines × chiffre à la position des unités) doit être minimisé.
solve minimize chiffres[3]*chiffres[2]*chiffres[4];

nombre = 1000 * chiffres[1] + 100 * chiffres[2] + 10 * chiffres[3]+ 1 * chiffres[4];

output [
  show(nombre),
  show(chiffres)
];


