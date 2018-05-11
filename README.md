Python module for plug based microfluidics data preprocessing
=============================================================

*  Tested in Python 3.6
*  Questions, issues: turei.denes@gmail.com

Usage:
-----

```python
    import plugy

     drugs = [
        '11:Void', '12:Void', '13:Nutlin', '14:Cyt-387', '15:IKK16',
        '16:MK-2206', '17:IL-6', '18:Gefitinib', '19:IFN-γ',
        '20:Soratinib', '21:TGF-β', '22:Dasatinib'
    ]

    p = plugy.Plugy(
        infile = 'example_screen.txt',
        cut = (3225, 11200),
        drugs = drugs
    )

    p.main()
```
