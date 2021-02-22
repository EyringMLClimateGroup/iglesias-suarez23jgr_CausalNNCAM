import sys, getopt, yaml, time, datetime
from datetime                     import datetime as dt
import numpy                          as np
from pathlib                      import Path
import utils.utils                    as utils
from   utils.constants            import DATA_FOLDER, ANCIL_FILE, experiment
#import utils.links                    as links
import utils.pcmci_algorithm          as algorithm


def single(
    gridpoints,
    var_parents,
    var_children,
    pc_alphas,
    parents_idx_levs,
    children_idx_levs,
    idx_lats,
    idx_lons,
    output_file_pattern,
    output_folder,
    overwrite
          ):
    
    ## Model's grid
    levels, latitudes, longitudes = utils.read_ancilaries(Path(DATA_FOLDER, ANCIL_FILE))
    
    ## Processing
    len_grid = len(gridpoints)
    t_start = time.time()
    for i_grid, (i_lat, i_lon) in enumerate(gridpoints):

        t_start_gridpoint = time.time()
        data_parents = None

        idx_lat = idx_lats[i_grid]
        idx_lon = idx_lons[i_grid]

        for child in var_children:
            print(f"{dt.now()} Variable: {child.name}")
            if child.dimensions == 2:
                child_levels = [[levels[-1],0]]
            elif child.dimensions == 3:
                child_levels = children_idx_levs
            for level in child_levels:

                results_filename = output_file_pattern.format(
                        var_name   = child.name,
                        level      = level[-1]+1,
                        lat        = int(i_lat),
                        lon        = int(i_lon),
                        experiment = experiment
                )
                results_file = Path(output_folder, results_filename)

                if not overwrite and results_file.is_file():
                    print(f"{dt.now()} Found file {results_file}, skipping.")
                    continue # Ignore this level


                # Only load parents if necessary to analyze a child
                # they stay loaded until the next gridpoint
                if data_parents is None:

                    print(f"{dt.now()} Gridpoint {i_grid+1}/{len_grid}: lat={latitudes[idx_lats[i_grid]]}"
                          + f" ({idx_lat}), lon={longitudes[idx_lons[i_grid]]} ({idx_lon})")

                    print(f"Load Parents (state fields)...")
                    t_before_load_parents = time.time()
                    data_parents = utils.load_data(
                        var_parents,
                        experiment,
                        DATA_FOLDER,
                        parents_idx_levs,
                        idx_lat,
                        idx_lon)
                    time_load_parents = datetime.timedelta(seconds = time.time() - t_before_load_parents)
                    print(f"{dt.now()} All parents loaded. Time: {time_load_parents}")

                # Process child
                data_child = utils.load_data([child],
                                       experiment,
                                       DATA_FOLDER,
                                       [level],
                                       idx_lat,
                                       idx_lon)
                data = [*data_parents, *data_child]

                # Find links
                print(f"{dt.now()} Finding links for {child.name} at level {level[-1]+1}")
                t_before_find_links = time.time()
                results = algorithm.find_links(data, pc_alphas, 0)
                time_links = datetime.timedelta(seconds = time.time() - t_before_find_links)
                total_time = datetime.timedelta(seconds = time.time() - t_start)
                print(f"{dt.now()} Links found. Time: {time_links}" + f" Total time so far: {total_time}")

                # Store causal links
                utils.save_results(results, results_filename, output_folder)

        time_point = datetime.timedelta(seconds = time.time() - t_start_gridpoint)
        total_time = datetime.timedelta(seconds = time.time() - t_start)
        print(f"{dt.now()} All links in gridpoint found. Time: {time_point}."
              + f" Total time so far: {total_time}")
        print("")

    print(f"{dt.now()} Execution complete. Total time: {total_time}")
    

    
def concat(
    gridpoints,
    var_parents,
    var_children,
    pc_alphas,
    parents_idx_levs,
    children_idx_levs,
    idx_lats,
    idx_lons,
    output_file_pattern,
    output_folder,
    overwrite
          ):
    
    ## Model's grid
    levels, latitudes, longitudes = utils.read_ancilaries(Path(DATA_FOLDER, ANCIL_FILE))
    
    ## Processing
    len_grid = len(gridpoints)
    t_start = time.time()
    
    ## outFile exists?
    for child in var_children:
        print(f"{dt.now()} Variable: {child.name}")
        if child.dimensions == 2:
            child_levels = [[levels[-1],0]]
        elif child.dimensions == 3:
            child_levels = children_idx_levs
        for level in child_levels:

            results_filename = output_file_pattern.format(
                    var_name   = child.name,
                    level      = level[-1]+1,
                    lat        = int(i_lat),
                    lon        = int(i_lon),
                    experiment = experiment
                )
            results_file = Path(output_folder, results_filename)

            if not overwrite and results_file.is_file():
                print(f"{dt.now()} Found file {results_file}, skipping.")
                continue # Ignore this level
    
            print('We should be here yet...')
            import pdb; pdb.set_trace()
    
    print('Shouldnt get to here yet...')
    import pdb; pdb.set_trace()
    
    
    for i_grid, (i_lat, i_lon) in enumerate(gridpoints):

        t_start_gridpoint = time.time()
        data_parents = None

        idx_lat = idx_lats[i_grid]
        idx_lon = idx_lons[i_grid]

        for child in var_children:
            print(f"{dt.now()} Variable: {child.name}")
            if child.dimensions == 2:
                child_levels = [[levels[-1],0]]
            elif child.dimensions == 3:
                child_levels = children_idx_levs
            for level in child_levels:

                results_filename = output_file_pattern.format(
                        var_name   = child.name,
                        level      = level[-1]+1,
                        lat        = int(i_lat),
                        lon        = int(i_lon),
                        experiment = experiment
                )
                results_file = Path(output_folder, results_filename)

                if not overwrite and results_file.is_file():
                    print(f"{dt.now()} Found file {results_file}, skipping.")
                    continue # Ignore this level


                # Only load parents if necessary to analyze a child
                # they stay loaded until the next gridpoint
                if data_parents is None:

                    print(f"{dt.now()} Gridpoint {i_grid+1}/{len_grid}: lat={latitudes[idx_lats[i_grid]]}"
                          + f" ({idx_lat}), lon={longitudes[idx_lons[i_grid]]} ({idx_lon})")

                    print(f"Load Parents (state fields)...")
                    t_before_load_parents = time.time()
                    data_parents = utils.load_data(
                        var_parents,
                        experiment,
                        DATA_FOLDER,
                        parents_idx_levs,
                        idx_lat,
                        idx_lon)
                    time_load_parents = datetime.timedelta(seconds = time.time() - t_before_load_parents)
                    print(f"{dt.now()} All parents loaded. Time: {time_load_parents}")

                # Process child
                data_child = utils.load_data([child],
                                       experiment,
                                       DATA_FOLDER,
                                       [level],
                                       idx_lat,
                                       idx_lon)
                data = [*data_parents, *data_child]

                # Find links
                print(f"{dt.now()} Finding links for {child.name} at level {level[-1]+1}")
                t_before_find_links = time.time()
                results = algorithm.find_links(data, pc_alphas, 0)
                time_links = datetime.timedelta(seconds = time.time() - t_before_find_links)
                total_time = datetime.timedelta(seconds = time.time() - t_start)
                print(f"{dt.now()} Links found. Time: {time_links}" + f" Total time so far: {total_time}")

                # Store causal links
                utils.save_results(results, 
                                   child, 
                                   level[-1], 
                                   i_lat, 
                                   i_lon, 
                                   experiment, 
                                   results_filename, 
                                   output_folder)

        time_point = datetime.timedelta(seconds = time.time() - t_start_gridpoint)
        total_time = datetime.timedelta(seconds = time.time() - t_start)
        print(f"{dt.now()} All links in gridpoint found. Time: {time_point}."
              + f" Total time so far: {total_time}")
        print("")

    print(f"{dt.now()} Execution complete. Total time: {total_time}")