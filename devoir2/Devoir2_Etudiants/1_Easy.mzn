%%%
% nom: Simon Kaplo           matricule: 1947701
% nom: Bassem Michel Ghaly   matricule: 1951389
%%%

include "globals.mzn";

enum ATTRACTION;

% Il faut sélectionner entre l et u attractions
int: l;
int: u;

% Attrait touristique total minimum
int: m;

% Attrait touristique par paire d'attraction
array[ATTRACTION, ATTRACTION] of int: pair_attraction;

% Coût de construction par attraction
array[ATTRACTION] of int: cost;

% Ensemble d'attractions à construire
var set of ATTRACTION: x;

% Coût total
var int: tot_cost;

% Le nombre d'attraction est entre l et u.
constraint l<=card(x) /\ card(x)<=u;
% La somme totale des couts d'attractions.
tot_cost = sum (a in x)(cost[a]);
% Attrait touristique total.
var int: m_tot = sum(a in x, b in x where a < b)(pair_attraction[a,b]);
% Attrait touristique total supérieur à m.
constraint m_tot > m;
% Minimise le cout totale.
solve minimize tot_cost;

output [
  "Attractions : ", show(x), "\n",
  "Cost : ", show(tot_cost)
];