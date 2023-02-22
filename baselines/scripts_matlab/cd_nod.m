function cd_nod(path_data, path_sig_level, dir_path)

disp('*** import_scripts... ');
scripts_matlab_root = fileparts(mfilename('fullpath'));
cd_nod_root = fullfile(scripts_matlab_root, 'matlab_packages', 'cd_nod');

addpath(cd_nod_root);
addpath(fullfile(cd_nod_root, 'KCI-test'));
addpath(fullfile(cd_nod_root, 'KCI-test', 'algorithms'));
addpath(fullfile(cd_nod_root, 'KCI-test', 'gpml-matlab', 'gpml'));

disp('*** preparing_input_data... ');
X = readtable(path_data, 'VariableNamingRule', 'preserve');
names = X.Properties.VariableNames;
X = X{:, :};

tmp = fileread(path_sig_level);
sig_level = str2double(tmp);

[time_step_count, variable_count] = size(X);

maxFanIn = 1;
cond_ind_test = 'indtest_new_t';
pars.pairwise = false;
pars.bonferroni = false;
pars.if_GP1 = 1;
pars.if_GP2 = 1;
pars.width = 0;
pars.widthT = 0.1;
c_indx = transpose([1:time_step_count]);
Type = 0;

disp('*** running_causal_discovery... ');
[~, ~, gns, ~] = nonsta_cd_new(X, cond_ind_test, c_indx, maxFanIn, sig_level, Type, pars);

disp('*** post_processing... ');
gns = gns(1:variable_count, 1:variable_count);

[connected_rows, connected_cols] = find(gns==-1);
[directed_rows, directed_cols] = find(gns==1);

for i = 1:length(connected_rows)
    gns(connected_rows(i), connected_cols(i)) = 1;
end

for i = 1:length(directed_rows)
    gns(directed_rows(i), directed_cols(i)) = 2;
end

disp('*** writing_table_to_file... ');
gns = array2table(gns,'VariableNames',names);
fprintf(dir_path+'/results/result.txt');
writetable(gns, './results/result.txt');

end
