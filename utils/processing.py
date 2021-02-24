import sys, getopt, yaml, time, datetime
from datetime                     import datetime as dt
import numpy                          as np
from pathlib                      import Path
import utils.utils                    as utils
from   utils.constants            import DATA_FOLDER, ANCIL_FILE, experiment
import utils.pcmci_algorithm          as algorithm


def single(
        gridpoints,
        var_parents,
        var_children,
        pc_alphas,
        levels,
        parents_idx_levs,
        children_idx_levs,
        idx_lats,
        idx_lons,
        output_file_pattern,
        output_folder,
        overwrite
):
    
    ## Processing
    len_grid = len(gridpoints)
    t_start = time.time()
    for i_grid, (lat, lon) in enumerate(gridpoints):

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
                results_file = utils.generate_results_filename(
                    child, level[-1], lat, lon, experiment,
                    output_file_pattern, output_folder)

                if not overwrite and results_file.is_file():
                    print(f"{dt.now()} Found file {results_file}, skipping.")
                    continue # Ignore this level


                # Only load parents if necessary to analyze a child
                # they stay loaded until the next gridpoint
                if data_parents is None:

                    print(f"{dt.now()} Gridpoint {i_grid+1}/{len_grid}:"
                          + f" lat={lat} ({idx_lat}), lon={lon} ({idx_lon})")

                    print(f"Load Parents (state fields)...")
                    t_before_load_parents = time.time()
                    data_parents = utils.load_data(
                        var_parents,
                        experiment,
                        DATA_FOLDER,
                        parents_idx_levs,
                        idx_lat,
                        idx_lon)
                    time_load_parents = datetime.timedelta(
                            seconds = time.time() - t_before_load_parents)
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
                time_links = datetime.timedelta(
                        seconds = time.time() - t_before_find_links)
                total_time = datetime.timedelta(seconds = time.time() - t_start)
                print(f"{dt.now()} Links found. Time: {time_links}"
                      + f" Total time so far: {total_time}")

                # Store causal links
                utils.save_results(results, results_file)

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
    levels,
    parents_idx_levs,
    children_idx_levs,
    idx_lats,
    idx_lons,
    output_file_pattern,
    output_folder,
    overwrite
          ):
    
    ## Processing
    len_grid     = len(gridpoints)
    t_start      = time.time()
    data_parents = None
    
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
                        lat1       = int(gridpoints[0][0]),
                        lat2       = int(gridpoints[-1][0]),
                        lon1       = int(gridpoints[0][-1]),
                        lon2       = int(gridpoints[-1][-1]),
                        experiment = experiment
                )
            results_file = Path(output_folder, results_filename)
    
    
            if not overwrite and results_file.is_file():
                print(f"{dt.now()} Found file {results_file}, skipping.")
                continue # Ignore this level
    

            # Only load parents if necessary to analyze a child
            # they stay loaded until the next gridpoint
            if data_parents is None:
                print(); print(f"Load Parents (state fields)...")
                t_before_load_parents = time.time()
                for i_grid, (lat, lon) in enumerate(gridpoints):

                    t_start_gridpoint = time.time()

                    idx_lat = idx_lats[i_grid]
                    idx_lon = idx_lons[i_grid]
                
                    print(f"{dt.now()} Gridpoint {i_grid+1}/{len_grid}:"
                          + f" lat={lat} ({idx_lat}), lon={lon} ({idx_lon})")

                    normalized_parents = utils.load_data_concat(
                            var_parents,
                            experiment,
                            DATA_FOLDER,
                            parents_idx_levs,
                            idx_lat,
                            idx_lon)
                    if data_parents is None:
                        data_parents = normalized_parents
                    else:
                        data_parents = np.concatenate((data_parents, normalized_parents), axis=1)
                print('Parents shape: ', data_parents.shape)
                # Format data
                data_parents = utils.format_data(data_parents, var_parents, parents_idx_levs)

                time_load_parents = datetime.timedelta(
                        seconds = time.time() - t_before_load_parents)
                print(f"{dt.now()} All parents loaded. Time: {time_load_parents}"); print("")
            
            
            # Process data child
            print(f"Load {child.name}...")
            t_before_load_child = time.time()
            data_child = None
            for i_grid, (lat, lon) in enumerate(gridpoints):
                
                idx_lat = idx_lats[i_grid]
                idx_lon = idx_lons[i_grid]

                print(f"{dt.now()} Gridpoint {i_grid+1}/{len_grid}:"
                      + f" lat={lat} ({idx_lat}), lon={lon} ({idx_lon})")
                
                normalized_child = utils.load_data_concat(
                        [child],
                        experiment,
                        DATA_FOLDER,
                        [level],
                        idx_lat,
                        idx_lon)
                if data_child is None:
                    data_child = normalized_child
                else:
                    data_child = np.concatenate((data_child, normalized_child), axis=1)
            print('Child shape: ', data_child.shape)
            time_load_child = datetime.timedelta(seconds = time.time() - t_before_load_child)
            print(f"{dt.now()} Child loaded. Time: {time_load_child}"); print("")
            
            # Format data
            data_child = utils.format_data(data_child, [child], [level])
            data = [*data_parents, *data_child]
            
            # Find links
            print(f"{dt.now()} Finding links for {child.name} at level {level[-1]+1}")
            t_before_find_links = time.time()
            results = algorithm.find_links(data, pc_alphas, 0)
            time_links = datetime.timedelta(seconds = time.time() - t_before_find_links)
            total_time = datetime.timedelta(seconds = time.time() - t_start)
            print(f"{dt.now()} Links found. Time: {time_links}" + f" Total time so far: {total_time}")
            print()
            
            # Store causal links
            utils.save_results(results, output_folder+'/'+results_filename)


    total_time = datetime.timedelta(seconds = time.time() - t_start)
    print(f"{dt.now()} Execution complete. Total time: {total_time}")