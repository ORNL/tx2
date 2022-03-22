Dashboard Interface
###################

This page contains brief descriptions of what the different buttons and checkboxes in the interface do.


UMAP Sidebar
============

The graph controls and status messages for the currently selected points are in the sidebar of the UMAP plot.
The top two status indicators respectively show the busy/ready status for backend computation and plotting work.
The model classification section shows the currently predicted label as well as the target (the colors match those used
in the plot itself.)

For the checkboxes under graph controls:

* Show training data - lightly includes the training points in the plot (larger and more transparent points), for helping determine if there's much difference between the training and testing distributions.
* Visual cluster nums - display the numerical labels for the computed clusters of the 2d projections. Theses labels are consistent with the plot titles in the cluster words bar graphs and wordclouds in the tabs below.
* Focus misclassifications - slightly blur out the points the transformer predicted correctly to make it easier to see which ones it got wrong.

The sample misclassified button will randomly select a point that the transformer got wrong.

The keyword search box allows you to type in a word and the plot will highlight in red the instances where that word appears.
The "sample from highlighted" button will randomly select a point from the highlighted instances.

Other controls
==============

The "selected datapoint index" below the UMAP plot is a dropdown menu containing the index of each point in the testing
set. Changing this selection will update the UMAP plot to highlight the current point, and the entry text and word salience
map will update.

In the Cluster words and Word clouds tabs, the "sampling buttons" will randomly select an instance from the visual cluster
of the corresponding label in square brackets on the button.
