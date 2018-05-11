Python module for plug based microfluidics data preprocessing
#############################################################

Usage:
======

.. code-block:: python
    
    import plugy

     drugs = [
        '11:Void', '12:Void', '13:Nutlin', '14:Cyt-387', '15:IKK16',
        '16:MK-2206', '17:IL-6', '18:Gefitinib', '19:IFN-γ',
        '20:Soratinib', '21:TGF-β', '22:Dasatinib'
    ]

    p = plugy.Plugy(infile = '06022018_mouse_kidney_1b.txt',
                        cut = (3225, 11200), drugs = drugs)

    p.main()
