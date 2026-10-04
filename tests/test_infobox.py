import unittest


class TestInfoboxExtraction(unittest.TestCase):
    def test_extract_infobox_fields(self):
        from src.ingestion.tigergraph_loader import TigerGraphLoader

        loader = TigerGraphLoader()

        # Valid infobox
        text1 = (
            "[Infobox Olympic event]\n"
            "games: 2016 Summer\n"
            "venue: Olympic Stadium\n"
            "event: 100m\n"
            "goldNOC: USA\n"
            "silverNOC: JAM\n"
            "bronzeNOC: GBR\n\n"
            "Later text games: 2020 Summer"
        )
        res1 = loader.extract_infobox_fields(text1)

        self.assertEqual(res1.get("games"), "2016 Summer")
        self.assertEqual(res1.get("venue"), "Olympic Stadium")
        self.assertEqual(res1.get("event"), "100m")
        self.assertEqual(res1.get("goldNOC"), "USA")
        self.assertEqual(res1.get("silverNOC"), "JAM")
        self.assertEqual(res1.get("bronzeNOC"), "GBR")

        # Fake fields later
        text2 = "Some prose games: 2020 Summer venue: Fake"
        res2 = loader.extract_infobox_fields(text2)
        self.assertEqual(res2, {})

        # Whitespace
        text3 = "[Infobox Olympic event]\ngames:   2008 Beijing   \n\n"
        res3 = loader.extract_infobox_fields(text3)

        self.assertEqual(res3.get("games"), "2008 Beijing")

    def test_flattened_infobox_q744487(self):
        from src.ingestion.tigergraph_loader import TigerGraphLoader

        loader = TigerGraphLoader()

        # Exact flattened Q744487-style text
        text = (
            "[Infobox Olympic event] "
            "event: Men's high jump "
            "games: 2008 Summer "
            "venue: Beijing Olympic Stadium "
            "dates: 17 August 2008 (qualifying)19 August 2008 (final) "
            "competitors: 40 "
            "nations: 28 "
            "longnames: yes "
            "gold: Andrey Silnov "
            "goldNOC: RUS "
            "silver: Germaine Mason "
            "silverNOC: GBR "
            "bronze: Yaroslav Rybakov "
            "bronzeNOC: RUS "
            "win_label: Winning height "
            "win_value: 2.36 "
            "prev: 2004 "
            "next: 2012 "
            "The men's high jump"
        )

        res = loader.extract_infobox_fields(text)

        self.assertEqual(res.get("games"), "2008 Summer")
        self.assertEqual(res.get("venue"), "Beijing Olympic Stadium")
        self.assertEqual(res.get("event"), "Men's high jump")
        self.assertEqual(res.get("goldNOC"), "RUS")
        self.assertEqual(res.get("silverNOC"), "GBR")
        self.assertEqual(res.get("bronzeNOC"), "RUS")