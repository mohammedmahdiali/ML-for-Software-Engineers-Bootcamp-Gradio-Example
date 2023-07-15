import gradio as gr
from functions.main_functions import *
from settings import *

with gr.Blocks() as brainable:
    with gr.Tab("Main") as main_tab:
        gr.Markdown(f"# <span style='color: #060c5e;'>{app_title}</span>")
        gr.Markdown(f"#### <div align='center'>{app_description}</div>")
        gr.Markdown(f"<div align='center'>Version: {app_version}</div>")
        logo = gr.Image(value=app_image, container=False, show_label=False)

    with gr.Tab("CNN Builder") as cnn_builder_tab:
        gr.Markdown(f"# <span style='color: #060c5e;'>CNN Builder</span>")

        with gr.Row():
            gr.Markdown("**Conv2D**:")
            conv2d_count = gr.Label(conv2d_layers_count, container=False, show_label=False)

            gr.Markdown("**MaxPooling2D**:")
            max_pooling_count = gr.Label(max_pooling_layers_count, container=False, show_label=False)

            gr.Markdown("**Dense**:")
            dense_count = gr.Label(dense_layers_count, container=False, show_label=False)

            gr.Markdown("**Status**:")
            action_status = gr.Label(container=False, show_label=False)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    gr.Markdown("Input Shape [32,32,3]:")
                    input_shape = gr.Textbox(container=False, show_label=False)
                with gr.Row():
                    gr.Markdown("Filters or Units (128):")
                    filters = gr.Number(container=False, show_label=False)
                with gr.Row():
                    gr.Markdown("Kernel or Pool Size (3,3):")
                    kernel_size = gr.Textbox(container=False, show_label=False)
                with gr.Row():
                    gr.Markdown("Activation (relu):")
                    activation = gr.Textbox(container=False, show_label=False)
                with gr.Row():
                    gr.Markdown("Padding (same):")
                    padding = gr.Textbox(container=False, show_label=False)
            
            with gr.Column():
                create_model_btn = gr.Button(value="Create Your Model")
                add_conv2d_btn = gr.Button(value="Add Conv2D")
                add_max_pooling_btn = gr.Button(value="Add MaxPooling2D")
                add_flatten_btn = gr.Button(value="Add Flatten")
                add_dense_btn = gr.Button(value="Add Dense")
                plot_arch_btn = gr.Button(value="Plot Architecture")

            with gr.Column():
                with gr.Row():
                    gr.Markdown("Optimizer (Adam):")
                    optimizer = gr.Textbox(container=False, show_label=False)

                with gr.Row():
                    gr.Markdown("Loss (CategoricalCrossentropy):")
                    loss = gr.Textbox(container=False, show_label=False)

                with gr.Row():
                    gr.Markdown("Metrics (','):")
                    metrics = gr.Textbox(container=False, show_label=False)

                with gr.Row():
                    epochs = gr.Slider(label="Epochs", interactive=True, maximum=2000)
                    validation_size = gr.Radio(choices=["5%", "10%", "20%", "30%"], label="Validation Size")

        with gr.Row():
            dataset = gr.File(label="Training Dataset")
            train_btn = gr.Button(value="Train")
            clear_btn = gr.Button(value="Clear Session")
            plot_history_btn = gr.Button(value="Plot History")

        with gr.Row():
            history_plot = gr.Plot(label="History")

    create_model_btn.click(
        fn=create_model,
        inputs=[input_shape],
        outputs=[action_status]
    )

    add_conv2d_btn.click(
        fn=add_conv2d,
        inputs=[filters, kernel_size, activation, padding],
        outputs=[action_status]
    )

    add_max_pooling_btn.click(
        fn=add_max_pooling,
        inputs=[kernel_size],
        outputs=[action_status]
    )

    add_flatten_btn.click(
        fn=add_flatten,
        inputs=None,
        outputs=[action_status]
    )

    add_dense_btn.click(
        fn=add_dense,
        inputs=[filters, activation],
        outputs=[action_status]
    )

    plot_arch_btn.click(
        fn=plot_architecture,
        inputs=[optimizer, loss, metrics],
        outputs=[action_status]
    )

    train_btn.click(
        fn=train_model,
        inputs=[dataset, epochs, validation_size],
        outputs=[action_status]
    )

    clear_btn.click(
        fn=delete_model,
        inputs=None,
        outputs=None
    )

    plot_history_btn.click(
        fn=plot_history,
        inputs=None,
        outputs=[history_plot]
    )
    
brainable.launch()
