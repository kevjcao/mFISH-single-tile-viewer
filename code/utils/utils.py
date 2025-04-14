# utils/my_utils.py
import os
import zarr
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from IPython.display import display


def extract_cell_id(zarr_paths):
    if not zarr_paths:
        raise ValueError("zarr_paths list is empty.")
    path = zarr_paths[0]
    parts = path.split("HCR_")[1]
    cell_id = parts.split("_")[0]
    for p in zarr_paths:
        if p.split("HCR_")[1].split("_")[0] != cell_id:
            raise ValueError("Inconsistent cell_id found in the dataset paths.")
    return cell_id

def compute_channel_defaults(store, channel_key, res_level, default_z_center, default_z_margin):
    ms_group = store[channel_key]
    level_array = ms_group[res_level]
    dask_img = da.from_array(level_array)
    z_min_default = max(0, default_z_center - default_z_margin)
    z_max_default = default_z_center + default_z_margin + 1
    img_slice = dask_img[0, 0, z_min_default:z_max_default, :, :].compute()
    img_proj = da.max(img_slice, axis=0).compute()
    gmin = 0
    gmax = float(np.max(img_proj) * 3) # max range is * 3 to increase dynamic range
    d_vmin, d_vmax = np.percentile(img_proj, [5, 95])
    return (gmin, gmax, d_vmin, d_vmax, img_proj)

def create_global_z_controls(total_z, default_z_center, default_z_margin, width="20%"):
    slider_z_center = widgets.IntSlider(
        value=default_z_center,
        min=0,
        max=total_z - 1,
        step=1,
        description='Z Center:',
        continuous_update=False,
        layout=widgets.Layout(width=width, margin='2px')
    )
    slider_z_margin = widgets.IntSlider(
        value=default_z_margin,
        min=0,
        max=total_z,
        step=1,
        description='Z Margin:',
        continuous_update=False,
        layout=widgets.Layout(width=width, margin='2px')
    )
    global_z_box = widgets.VBox([widgets.Label("Global Z Controls:", layout=widgets.Layout(margin='2px')),
                                 slider_z_center, slider_z_margin],
                                 layout=widgets.Layout(margin='2px', padding='2px'))
    return slider_z_center, slider_z_margin, global_z_box

def create_channel_control_box(ch, defaults, gene, width="400px"):
    # defaults is a tuple: (gmin, gmax, def_vmin, def_vmax)
    gmin, gmax, def_vmin, def_vmax = defaults
    slider_vmin = widgets.FloatSlider(
        value=def_vmin,
        min=gmin,
        max=gmax,
        step=(gmax - gmin) / 100.0 if gmax != gmin else 1,
        description='',
        continuous_update=False,
        layout=widgets.Layout(width='60%', margin='2px')
    )
    text_vmin = widgets.FloatText(
        value=def_vmin,
        layout=widgets.Layout(width='60px', margin='2px')
    )
    widgets.jslink((slider_vmin, 'value'), (text_vmin, 'value'))
    min_box = widgets.HBox([widgets.Label("Min:", layout=widgets.Layout(width='40px')), slider_vmin, text_vmin],
                           layout=widgets.Layout(margin='2px'))
    
    slider_vmax = widgets.FloatSlider(
        value=def_vmax,
        min=gmin,
        max=gmax,
        step=(gmax - gmin) / 100.0 if gmax != gmin else 1,
        description='',
        continuous_update=False,
        layout=widgets.Layout(width='60%', margin='2px')
    )
    text_vmax = widgets.FloatText(
        value=def_vmax,
        layout=widgets.Layout(width='60px', margin='2px')
    )
    widgets.jslink((slider_vmax, 'value'), (text_vmax, 'value'))
    max_box = widgets.HBox([widgets.Label("Max:", layout=widgets.Layout(width='40px')), slider_vmax, text_vmax],
                           layout=widgets.Layout(margin='2px'))
    
    # color picker for this channel (default white)
    color_picker = widgets.ColorPicker(
        value='#ffffff',
        description='Color:',
        layout=widgets.Layout(width='60%', margin='2px')
    )
    
    control_box = widgets.VBox([widgets.Label(f"Channel {ch} ({gene})", layout=widgets.Layout(margin='2px')),
                                min_box, max_box, color_picker],
                               layout=widgets.Layout(margin='2px', padding='2px', width=width))
    return control_box, slider_vmin, slider_vmax, color_picker

def update_channel_plot(ch, store, res_level, z_center, z_margin, vmin, vmax, color, gene, dot_x, dot_y, dot_on, dot_size, dot_color):
    z_min_current = max(0, int(z_center) - int(z_margin))
    z_max_current = int(z_center) + int(z_margin) + 1
    try:
        channel_key = f"Tile_X_0000_Y_0000_Z_0000_ch_{ch}.zarr"
        ms_group = store[channel_key]
        level_array = ms_group[res_level]
        dask_img = da.from_array(level_array)
        img_slice = dask_img[0, 0, z_min_current:z_max_current, :, :].compute()
        img_proj = da.max(img_slice, axis=0).compute()
    except Exception as e:
        print(f"Error loading channel {ch}: {e}")
        img_proj = None

    custom_cmap = LinearSegmentedColormap.from_list('custom', ['black', color])
    plt.figure(figsize=(6, 6))
    if img_proj is not None:
        plt.imshow(img_proj, cmap=custom_cmap, vmin=vmin, vmax=vmax)
        plt.title(f"Channel {ch} ({gene})\n(Z: {z_min_current} to {z_max_current-1})", fontsize=10)
        if dot_on:
            # place a dot at the specified (dot_x, dot_y) with specified dot color and size
            plt.plot(dot_x, dot_y, 'o', markersize=dot_size, markeredgecolor=dot_color, markerfacecolor='none', linewidth=2)
    else:
        plt.text(0.5, 0.5, 'Error', ha='center', va='center')
    plt.axis('off')
    plt.show()

def create_round_interactive_view(round_idx, main_channel, channels_per_round, genes_list, res_level, zarr_paths, default_z_center, default_z_margin):
    """
    Create an interactive viewer for a specified round, plots each channel in the selected round.
    Displays global Z-controls (brightness sliders, text inputs, color picker) and dot controls (x, y positions, dot size and color). 
        - Can also toggle dot on/off.
    """
    store = zarr.open(zarr_paths[round_idx], mode='r')
    round_channels = channels_per_round[round_idx]
    gene_dict = genes_list[round_idx]
    
    # get total number of Z-planes from main ch
    channel_key_main = f"Tile_X_0000_Y_0000_Z_0000_ch_{main_channel}.zarr"
    ms_group_main = store[channel_key_main]
    level_array_main = ms_group_main[res_level]
    dask_img_main = da.from_array(level_array_main)
    total_z = dask_img_main.shape[2]
    
    # get brightness defaults for each ch
    defaults = {}
    for ch in round_channels:
        try:
            channel_key = f"Tile_X_0000_Y_0000_Z_0000_ch_{ch}.zarr"
            ms_group = store[channel_key]
            level_array = ms_group[res_level]
            dask_img = da.from_array(level_array)
            z_min_default = max(0, default_z_center - default_z_margin)
            z_max_default = default_z_center + default_z_margin + 1
            img_slice = dask_img[0, 0, z_min_default:z_max_default, :, :].compute()
            img_proj = da.max(img_slice, axis=0).compute()
            gmin = 0
            gmax = float(np.max(img_proj) * 3)
            d_vmin, d_vmax = np.percentile(img_proj, [5, 95])
            defaults[ch] = (gmin, gmax, d_vmin, d_vmax)
        except Exception as e:
            print(f"Error computing defaults for channel {ch}: {e}")
            defaults[ch] = (0, 1e4, 0, 1e4)
            
    # global Z-controls
    slider_z_center, slider_z_margin, global_z_box = create_global_z_controls(total_z, default_z_center, default_z_margin, width="80%")
    z_center_text = widgets.IntText(value=slider_z_center.value, layout=widgets.Layout(width='60px', margin='2px'))
    z_margin_text = widgets.IntText(value=slider_z_margin.value, layout=widgets.Layout(width='60px', margin='2px'))
    widgets.jslink((slider_z_center, 'value'), (z_center_text, 'value'))
    widgets.jslink((slider_z_margin, 'value'), (z_margin_text, 'value'))
    z_center_box = widgets.HBox([slider_z_center, z_center_text], layout=widgets.Layout(margin='2px', padding='2px'))
    z_margin_box = widgets.HBox([slider_z_margin, z_margin_text], layout=widgets.Layout(margin='2px', padding='2px'))
    z_controls_combined = widgets.VBox([widgets.Label("Z Controls:"), z_center_box, z_margin_box],
                                         layout=widgets.Layout(width='50%', margin='2px', padding='2px'))
    
    # dot controls
    vol_shape = dask_img_main[0, 0, ...].shape  # (z, height, width)
    default_dot_x = vol_shape[2] / 2.0
    default_dot_y = vol_shape[1] / 2.0
    dot_x_input = widgets.FloatText(value=default_dot_x, layout=widgets.Layout(width='100px', margin='2px'))
    dot_y_input = widgets.FloatText(value=default_dot_y, layout=widgets.Layout(width='100px', margin='2px'))
    dot_checkbox = widgets.Checkbox(value=False, description='Show dot', layout=widgets.Layout(margin='2px'))
    dot_size_slider = widgets.FloatSlider(value=10, min=1, max=50, step=1, # update max as needed 
                                          continuous_update=False, layout=widgets.Layout(width='350px', margin='2px'))
    dot_size_row = widgets.HBox([
        widgets.Label("Dot size:", layout=widgets.Layout(width='70px')),
        dot_size_slider
    ], layout=widgets.Layout(align_items='center', justify_content='flex-start'))
    
    dot_color_picker = widgets.ColorPicker(value='#ff0000', 
                                           layout=widgets.Layout(width='50%', margin='2px'))
    dot_color_row = widgets.HBox([
        widgets.Label("Dot color:", layout=widgets.Layout(width='70px')),
        dot_color_picker
    ], layout=widgets.Layout(align_items='center', justify_content='flex-start'))
    
    dot_controls_box = widgets.VBox([
        widgets.Label("Dot Controls:"),
        widgets.HBox([
            widgets.Label("X:", layout=widgets.Layout(width='20px')),
            dot_x_input,
            widgets.Label("Y:", layout=widgets.Layout(width='20px')),
            dot_y_input
        ], layout=widgets.Layout(justify_content='flex-start')),
        dot_checkbox,
        dot_size_row,
        dot_color_row
    ], layout=widgets.Layout(width='25%', align_items='flex-start', margin='2px', padding='2px'))
    
    global_controls_box = widgets.HBox([z_controls_combined, dot_controls_box],
                                         layout=widgets.Layout(justify_content='flex-start', width='80%', gap='10px'))
    
    # control boxes for each ch
    channel_boxes = []
    for ch in round_channels:
        gene = gene_dict.get(ch, "NA")
        control_box, slider_vmin, slider_vmax, color_picker = create_channel_control_box(ch, defaults[ch], gene, width="400px")
        out = widgets.interactive_output(
            lambda vmin, vmax, z_center, z_margin, color, dot_x, dot_y, dot_on, dot_size, dot_color, ch=ch, gene=gene:
                update_channel_plot(ch, store, res_level, z_center, z_margin, vmin, vmax, color, gene, dot_x, dot_y, dot_on, dot_size, dot_color),
            {
                'vmin': slider_vmin,
                'vmax': slider_vmax,
                'z_center': slider_z_center,
                'z_margin': slider_z_margin,
                'color': color_picker,
                'dot_x': dot_x_input,
                'dot_y': dot_y_input,
                'dot_on': dot_checkbox,
                'dot_size': dot_size_slider,
                'dot_color': dot_color_picker
            }
        )
        box = widgets.VBox([control_box, out],
                           layout=widgets.Layout(margin='2px', padding='2px', width="600px"))
        channel_boxes.append(box)
    
    channels_hbox = widgets.HBox(channel_boxes, layout=widgets.Layout(justify_content='flex-start', margin='2px', padding='2px'))
    display(global_controls_box, channels_hbox)


def create_multi_round_view(target_channel, zarr_paths, channels_per_round, res_level, default_z_center, default_z_margin):
    """
    Create a subplot for a specific ch across rounds; plot the max intensity projection.
    """
    round_outputs = []
    z_min_default = max(0, default_z_center - default_z_margin)
    z_max_default = default_z_center + default_z_margin + 1
    for round_idx, zarr_path in enumerate(zarr_paths):
        if target_channel not in channels_per_round[round_idx]:
            print(f"Channel {target_channel} not found for round {round_idx+1}.")
            continue
        store = zarr.open(zarr_path, mode='r')
        channel_key = f"Tile_X_0000_Y_0000_Z_0000_ch_{target_channel}.zarr"
        try:
            ms_group = store[channel_key]
            level_array = ms_group[res_level]
            dask_img = da.from_array(level_array)
            img_slice = dask_img[0, 0, z_min_default:z_max_default, :, :].compute()
            img_proj = da.max(img_slice, axis=0).compute()
        except Exception as e:
            print(f"Error loading channel {target_channel} for round {round_idx+1}: {e}")
            continue
        
        global_min = float(np.min(img_proj))
        global_max = float(np.max(img_proj))
        default_vmin, default_vmax = np.percentile(img_proj, [5, 95])
        
        slider_vmin = widgets.FloatSlider(
            value=default_vmin,
            min=global_min,
            max=global_max,
            step=(global_max - global_min) / 100.0 if global_max != global_min else 1,
            description='Min:',
            continuous_update=False,
            layout=widgets.Layout(width='80%')
        )
        text_vmin = widgets.FloatText(
            value=default_vmin,
            layout=widgets.Layout(width='20%')
        )
        widgets.jslink((slider_vmin, 'value'), (text_vmin, 'value'))
        
        slider_vmax = widgets.FloatSlider(
            value=default_vmax,
            min=global_min,
            max=global_max,
            step=(global_max - global_min) / 100.0 if global_max != global_min else 1,
            description='Max:',
            continuous_update=False,
            layout=widgets.Layout(width='80%')
        )
        text_vmax = widgets.FloatText(
            value=default_vmax,
            layout=widgets.Layout(width='20%')
        )
        widgets.jslink((slider_vmax, 'value'), (text_vmax, 'value'))
        
        def update_round_image(vmin, vmax, img_proj=img_proj, round_idx=round_idx):
            plt.figure(figsize=(6, 6))
            plt.imshow(img_proj, cmap='magma', vmin=vmin, vmax=vmax)
            plt.title(f"Round {round_idx+1} - Channel {target_channel}")
            plt.axis('off')
            plt.show()
        
        out = widgets.interactive_output(update_round_image, {'vmin': slider_vmin, 'vmax': slider_vmax})
        
        round_box = widgets.VBox([
            widgets.Label(value=f"Round {round_idx+1} - Channel {target_channel}"),
            widgets.HBox([slider_vmin, text_vmin]),
            widgets.HBox([slider_vmax, text_vmax]),
            out
        ], layout=widgets.Layout(margin='2px', padding='2px', width="400px"))
        
        round_outputs.append(round_box)
    if round_outputs:
        rounds_hbox = widgets.HBox(round_outputs, layout=widgets.Layout(justify_content='center'))
        display(rounds_hbox)

def create_3d_view(round_idx, target_channel, zarr_paths, res_level_3d):
    """
    Create an interactive 3D view for a specified round and channel.
    """
    store = zarr.open(zarr_paths[round_idx], mode='r')
    channel_key = f"Tile_X_0000_Y_0000_Z_0000_ch_{target_channel}.zarr"
    ms_group = store[channel_key]
    dask_img = da.from_array(ms_group[res_level_3d])
    volume = dask_img[0, 0, ...].compute()
    print("Volume shape:", volume.shape)
    
    def update_3d(threshold):
        mask = volume > threshold
        z_idx, y_idx, x_idx = np.nonzero(mask)
        intensities = volume[z_idx, y_idx, x_idx]
        max_points = 10000
        if len(x_idx) > max_points:
            indices = np.linspace(0, len(x_idx)-1, max_points).astype(int)
            x_idx = x_idx[indices]
            y_idx = y_idx[indices]
            z_idx = z_idx[indices]
            intensities = intensities[indices]
        fig3d = go.Figure(data=[go.Scatter3d(
            x=x_idx,
            y=y_idx,
            z=z_idx,
            mode='markers',
            marker=dict(
                size=2,
                color=intensities,
                colorscale='Magma',
                opacity=0.8,
                colorbar=dict(title='Intensity')
            )
        )])
        fig3d.update_layout(
            title=f"3D View of Channel {target_channel} (Res {res_level_3d})",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        fig3d.show()
    
    slider_threshold = widgets.FloatSlider(
        value=1000,
        min=0,
        max=float(np.max(volume)),
        step=float(np.max(volume)) / 100.0,
        description='Threshold:',
        continuous_update=False,
        layout=widgets.Layout(width='80%')
    )
    
    # display the interactive plot
    display(slider_threshold)
    output = widgets.interactive_output(update_3d, {'threshold': slider_threshold})
    display(output)
