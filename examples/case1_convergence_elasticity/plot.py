import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=25)

params = {
    "text.latex.preamble": r"\usepackage{bm,amsmath,siunitx}\newcommand{\sib}[1]{[\si{#1}]}"
}
plt.rcParams.update(params)
mpl.rcParams["axes.linewidth"] = 1.5


def plot_error(blocks_all, data_file_all, out_file, linestyle_all, h_pos=2):
    folder = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots()

    h_all = []
    for blocks, data_file, linestyle in zip(blocks_all, data_file_all, linestyle_all):
        file_name = os.path.join(folder, data_file)
        data = np.loadtxt(file_name)

        for name, block in blocks.items():
            ax.loglog(
                data[block, h_pos],
                data[block, 0],
                label=name,
                marker="o",
                linestyle=linestyle,
            )

        h_all.append(data[:, h_pos])

    h_all = np.concatenate(h_all)
    h = [np.amin(h_all), np.amax(h_all)]
    ax.loglog(
        h,
        2 * np.power(h, 2),
        "-.",
        alpha=0.5,
        color="black",
    )
    pos_h = np.average(h)
    ax.text(pos_h, 1.3 * np.power(pos_h, 2), "$\mathcal{O}(h^2)$", fontsize=20)

    ax.set_xlabel("$h$")
    ax.set_title("$L^2$-error $u$")
    # plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=10)

    file_name = os.path.join(folder, out_file) + ".pdf"
    fig.savefig(file_name, bbox_inches="tight")

    # plot the legend
    handles, labels = [
        (a + b)
        for a, b in zip(ax.get_legend_handles_labels(), ax.get_legend_handles_labels())
    ]
    reorder = [0, 3, 2, 1, 5, 6, 4]
    labels, mask = np.unique(labels, return_index=True)
    labels = [labels[i] for i in reorder]
    handles = [handles[i] for i in mask[reorder]]

    fig, ax = plt.subplots(figsize=(25, 10))
    for h, l in zip(handles, labels):
        ax.plot(np.zeros(1), label=l)

    ax.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(-0.1, -0.65))
    file_name = os.path.join(folder, out_file) + "_legend.pdf"
    fig.savefig(file_name, bbox_inches="tight")

    os.system("pdfcrop --margins '0 -750 0 0' " + file_name + " " + file_name)
    os.system("pdfcrop " + file_name + " " + file_name)


def plot_cond(blocks_all, data_file_all, out_file, linestyle_all, h_pos=2):
    folder = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots()

    for blocks, data_file, linestyle in zip(blocks_all, data_file_all, linestyle_all):
        file_name = os.path.join(folder, data_file)
        data = np.loadtxt(file_name)

        for name, block in blocks.items():
            ax.loglog(
                data[block, h_pos],
                data[block, 4],
                label=name,
                marker="o",
                linestyle=linestyle,
            )

    ax.set_xlabel("$h$")
    ax.set_title("condition number")

    # plt.legend(loc="upper left")

    file_name = os.path.join(folder, out_file) + ".pdf"
    fig.savefig(file_name, bbox_inches="tight")
    plt.gcf().clear()


if __name__ == "__main__":

    blocks_1 = {
        "Monotile": slice(0, 4),
        "Voronoi": slice(4, 8),
        "Simplices": slice(8, 12),
    }
    linestyle_1 = "-"

    blocks_2 = {
        "Monotile reg1": slice(0, 4),
        "Voronoi reg1": slice(4, 8),
        "Voronoi reg2": slice(8, 12),
        "Voronoi Lloyd": slice(12, 16),
    }
    linestyle_2 = "--"

    plot_error(
        (blocks_1, blocks_2),
        ("err_1.txt", "err_2.txt"),
        "err",
        (linestyle_1, linestyle_2),
    )

    plot_cond(
        (blocks_1, blocks_2),
        ("err_1.txt", "err_2.txt"),
        "cond",
        (linestyle_1, linestyle_2),
    )
