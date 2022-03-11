Orbit Determination Laboratory, for exploring and developing novel orbit determination and wrapping existing orbit determination packages for comparisons

Feature list
-------------

* Monte-Carlo Markov-Chain evaluation of posterior distribution
* Experimental sensor-fusion methods
* Wrapps other orbit-determination software in a conventient manner


Install
--------

.. code-block:: bash

   git clone https://github.com/danielk333/odlab.git
   cd odlab
   pip install .


Testing
--------

You need 

* pytest >= 5.3.5

.. code-block:: bash

   pytest

If you want to test orekit propagator, that requires data to function, call pytest with the additional parameter `orekit_data`. Otherwise these tests are skipped.

.. code-block:: bash

   pytest --orekit_data='path/to/orekit-data-master.zip'


Build documentation
--------------------

You need 

* sphinx >= 3.0.3

.. code-block:: bash

   cd docs; make html


To reference
--------------

Please contact us (daniel.kastinen@irf.se) before using for publication or projects, this is an experimental repository and may need information from the code-authors about stability and other concerns.