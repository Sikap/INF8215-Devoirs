include "globals.mzn";

enum PROJECT;
enum WORKER = { ARCHI, ING, ELEC };

int: P; % nombre de jours (et de projets à choisir)

array[PROJECT, PROJECT] of int: travel_costs;     % Couts de deplacements entre projets

set of PROJECT: project_with_precedences;         % Ensemble des projets ayant des contraintes de précédences
array[PROJECT, 1..3] of WORKER: precedences;      % Précédences (pour les projets n'ayant pas de contraintes de précédences, ignorez la rangée)

set of PROJECT: project_must_be_done;             % Ensemble des projets qui DOIVENT êtres faits

array[PROJECT] of int: revenu;                    % Revenu des projets

array[WORKER, 1..P] of var PROJECT: schedule;     % Pour chaque employé, pour chaque jour, on assigne un projet
array[WORKER, PROJECT] of var 0..P: day;          % Pour chaque employé, pour chaque projet, on détermine le jour 

% Profit (gain - dépenses) a maximiser
var int: profit;
var int: gain;
var int: depenses;
var set of PROJECT: finished_project;
 

% Constraint: il ne peut pas y avoir plus d’un employé assigné à un projet par jour.
constraint forall (jour in 1..P)(alldifferent([schedule[i,jour]| i in WORKER]));
% Constraint: établis un lien entre le jour et le projet de l'employé.
constraint forall (worker in WORKER,jour in 1..P)(day[worker,schedule[worker,jour]]=jour);
% Constraint: un employé peut qu'une seul fois etre assigné a un projet. 
constraint forall (worker in WORKER)(alldifferent([day[worker,project]| project in PROJECT where day[worker,project]>0]));
% Constraint: un projet est seulement terminer lorsque l’ingénieur, l’architecte et l’électricien ont chacun travaillé une journée dessus.
constraint  forall(worker in WORKER,project in finished_project)(day[worker,project]>0) ;
% Constraint: les projets qui doivent absolument être complétés doit faire parti de se terminer.
constraint finished_project =  finished_project union project_must_be_done ;
% Constraint: les projet terminer doive prendre en consideration l’ordre de passage des employés.
constraint forall(project in project_with_precedences intersect finished_project)(day[precedences[project,1],project]<day[precedences[project,2],project] /\ day[precedences[project,2],project]<day[precedences[project,3],project]); 
% Les gains corespond a la somme des revenu des projet terminer.
gain = sum(projet in finished_project) (revenu[projet]);
% Les dépenses  correspond a la somme des cout de transfert des employés.
depenses = sum(worker in WORKER,jour in 1..P-1)(travel_costs[schedule[worker,jour],schedule[worker,jour+1]]);
% Profit (gain - dépenses) a maximiser.
profit = gain - depenses;
solve maximize profit;

output

["Horaire : \n"] 
++
[if p = 1 then show(w) ++ ":" else "" endif ++ show(schedule[w,p]) ++ if p = P then "\n" else "," endif | w in WORKER, p in 1..P]
++
["-----------\n"]
++
["Profit : " ++ show(profit)]
    