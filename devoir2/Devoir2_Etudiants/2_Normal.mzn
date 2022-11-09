include "globals.mzn";

% Nombre de segments horizontaux et verticaux
int: h; % nb de rows
int: v; % nb de cols

% La demande par segment horizontal et vertical
array[1..h] of int: demand_per_h;
array[1..v] of int: demand_per_v;

% Le coût du permis et de construction de maison associé à chaque carré
array[1..h, 1..v] of int: cost_per_house;
array[1..h, 1..v] of int: cost_per_permit;

% Le nombre max de maison qui peut être construit par carré
int: m;

% Le nombre de maisons à construire par carré
array[1..h, 1..v] of var int: x;

% Le coût total
var int: tot_cost;

% Nombre de maisons à construire plus grand que la demande horizentale.
constraint forall(h_i in 1..h)(sum(v_i in 1..v)(x[h_i, v_i]) >= demand_per_h[h_i]);
% Nombre de maisons à construire plus grand que la demande verticale. 
constraint forall(v_i in 1..v)(sum(h_i in 1..h)(x[h_i, v_i]) >= demand_per_v[v_i]);
% Nombre de maisons à construire dans un pâté inferieur a m. 
constraint forall(h_i in 1..h, v_i in 1..v)(x[h_i, v_i] <= m);
% Nombre de maisons à construire dans un pâté positif ou nulle. 
constraint forall(h_i in 1..h, v_i in 1..v)(x[h_i, v_i] >= 0);
% Cout totale de construction de tous les pâtés.
constraint tot_cost = sum(h_i in 1..h, v_i in 1..v)(if x[h_i,v_i]>0 then x[h_i, v_i]*cost_per_house[h_i, v_i] + cost_per_permit[h_i, v_i] else 0 endif);

solve minimize tot_cost;

output
  
  ["Configuration : \n"]  
  
  ++
  
  [
    show(x[i,j]) ++ if j = v then "\n" else "," endif 
    | i in 1..h, j in 1..v
  ]
  
  ++
  
  ["----------------\n"]
  
  ++
  
  [
  "Cost : " ++ show(tot_cost)
  ]
  