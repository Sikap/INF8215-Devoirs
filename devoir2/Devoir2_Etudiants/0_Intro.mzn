%%%
% nom: Simon Kaplo           matricule: 1947701
% nom: Bassem Michel Ghaly   matricule: 1951389
%%%

include "globals.mzn";

var int: nombre;
array[1..4] of var 1..9: chiffres;

constraint nombre = 1000*chiffres[1] + 100*chiffres[2] + 10*chiffres[3] + 1*chiffres[4];

% C’est un nombre pair.
constraint chiffres[4] mod 2 = 0;
% Le chiffre 0 n’est pas présent dans le nombre.
constraint forall(i in 1..4)(chiffres[i]>0);
% Les 4 chiffres sont différents.
constraint all_different(chiffres);
% Le chiffre à la position des milliers est supérieur à celui à la position des centaines.
constraint chiffres[1] > chiffres[2];
% Le chiffre à la position des dizaines est inférieur à celui à la position des unités.
constraint chiffres[3] < chiffres[4];
% Le chiffre à la position des centaines est supérieur à celui à la position des unités.
constraint chiffres[2] > chiffres[4];
% La somme des 4 chiffres est supérieure à 15.
constraint sum(chiffres) > 15 ;
% Le produit des 3 derniers chiffres (chiffre à la position des centaines × chiffre à la position des dizaines × chiffre à la position des unités) doit être minimisé.
solve minimize chiffres[3]*chiffres[2]*chiffres[4];

output [
  show(nombre),
];


