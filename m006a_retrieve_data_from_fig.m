for filename = {'/storage/data_2020-02-03/Eloss_DEH1.fig', '/storage/data_2020-02-03/Eloss_DEH2.fig', '/storage/data_2020-02-03/Eloss_DEH1-2.fig'}
    fig = openfig(char(filename));

    dataObjs = fig.Children.Children;
    all_x = findobj(fig, '-property', 'XData');

    outp = struct;

    for n_obj = 1:length(all_x)
        this_x = all_x(n_obj);
        this_x.XData;
        name = strrep(this_x.DisplayName, ' ', '_');
        for attr = {'XData', 'YData', 'YPositiveDelta', 'YNegativeDelta'}
            eval_str = sprintf('outp.%s_%s = this_x.%s;', name, char(attr), char(attr));
            eval(eval_str)

        end

    end

    save(sprintf('%s.mat', char(filename)), '-struct', 'outp')
end