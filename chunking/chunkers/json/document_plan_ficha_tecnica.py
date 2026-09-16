from chunking.chunkers.json.JSON_chunker import SplitNode

DOCUMENT_PLAN_FC = SplitNode(
    name="complete_document",
    children=[
        # identification: medicine_name, composition, pharmaceutical_form, indications 
        SplitNode(
            name="identification",
            children=[
                SplitNode(
                    name="medicine_name_composition_pharmaceutical_form",
                    children=[
                        SplitNode(
                            name="medicine_name_pharmaceutical_form",
                            children=[
                                SplitNode(
                                    name="medicine_name",
                                    fields=["medicine_name"]
                                ),
                                SplitNode(
                                    name="pharmaceutical_form",
                                    fields=["pharmaceutical_form"]
                                )
                            ]
                        ),
                        SplitNode(
                            name="composition",
                            fields=["composition"]
                        )
                    ]
                ),
                SplitNode(
                    name="indications",
                    fields=["indications"]
                )
            ]
        ),

        # posology
        SplitNode(
            name="posology",
            fields=["posology"]
        ),

        # contraindications
        SplitNode(
            name="contraindications",
            fields=["contraindications"]
        ),

        # warnings
        SplitNode(
            name="warnings_and_precautions",
            fields=["warnings_and_precautions"]
        ),

        # interactions
        SplitNode(
            name="interactions",
            fields=["interactions"]
        ),

        # pregnancy
        SplitNode(
            name="pregnancy_and_lactation",
            fields=["pregnancy_and_lactation"]
        ),

        # driving
        SplitNode(
            name="effects_on_driving",
            fields=["effects_on_driving"]
        ),

        # adverse_reactions
        SplitNode(
            name="adverse_reactions",
            fields=["adverse_reactions"]
        ),

        # overdose
        SplitNode(
            name="overdose",
            fields=["overdose"]
        ),

        # pharmacodynamics
        SplitNode(
            name="pharmacodynamics",
            fields=["pharmacodynamics"]
        ),

        # pharmacokinetics
        SplitNode(
            name="pharmacokinetics",
            fields=["pharmacokinetics"]
        ),

        # pharmaceutical
        SplitNode(
            name="pharmaceutical",
            children=[
                SplitNode(
                    name="excipients",
                    fields=["excipients"]
                ),
                SplitNode(
                    name="incompatibilities_shelf_life",
                    children=[
                        SplitNode(
                            name="incompatibilities",
                            fields=["incompatibilities"]
                        ),
                        SplitNode(
                            name="shelf_life",
                            fields=["shelf_life"]
                        )
                    ]
                ),
                SplitNode(
                    name="storage_container_handling",
                    children=[
                        SplitNode(
                            name="storage_container",
                            children=[
                                SplitNode(
                                    name="storage_conditions",
                                    fields=["storage_conditions"]
                                ),
                                SplitNode(
                                    name="container",
                                    fields=["container"]
                                )
                            ]
                        ),
                        SplitNode(
                            name="handling_instructions",
                            fields=["handling_instructions"]
                        )
                    ]
                )
            ]
        ),

        # marketing: marketing_authorisation_holder, marketing_authorisation_number
        SplitNode(
            name="marketing",
            children=[
                SplitNode(
                    name="marketing_authorisation_holder",
                    fields=["marketing_authorisation_holder"]
                ),
                SplitNode(
                    name="marketing_authorisation_number",
                    fields=["marketing_authorisation_number"]
                )
            ]
        ),

        # administration: active_ingredient, strength, route_of_administration
        SplitNode(
            name="administration",
            children=[
                SplitNode(
                    name="active_ingredient_strength",
                    children=[
                        SplitNode(
                            name="active_ingredient",
                            fields=["active_ingredient"]
                        ),
                        SplitNode(
                            name="strength",
                            fields=["strength"]
                        )
                    ]
                ),
                SplitNode(
                    name="route_of_administration",
                    fields=["route_of_administration"]
                )
            ]
        ),

        # national codes
        SplitNode(
            name="national_codes",
            fields=["national_codes"]
        ),

        # source
        SplitNode(
            name="_source",
            fields=["_source"]
        ),

        # reimbursement
        SplitNode(
            name="reimbursement",
            fields=["reimbursement"]
        )
    ]
)