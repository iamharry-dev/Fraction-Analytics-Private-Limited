import React, { useState, useEffect } from "react";
import { makeStyles } from "@material-ui/core/styles";
import {
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    TablePagination,
    TableFooter,
    Typography,
} from "@material-ui/core";
import jsonData from "./Records.json";

const useStyles = makeStyles((theme) => ({
    table: {
        minWidth: 650,
    },
    tableContainer: {
        borderRadius: 15,
        margin: "10px auto 10px auto",
        maxWidth: "95%",
    },
    tableHeaderCell: {
        fontWeight: "bold",
        backgroundColor: theme.palette.primary.dark,
        color: theme.palette.getContrastText(theme.palette.primary.dark),
    },
    avatar: {
        backgroundColor: theme.palette.primary.light,
        color: theme.palette.getContrastText(theme.palette.primary.light),
    },
    name: {
        fontWeight: "bold",
        color: theme.palette.secondary.dark,
    },
    status: {
        fontWeight: "bold",
        fontSize: "0.75rem",
        borderRadius: 8,
        padding: "3px 10px",
        display: "inline-block",
    },
    entryStatus: {
        backgroundColor: "green",
        color: "white",
    },
    exitStatus: {
        backgroundColor: "red",
        color: "white",
    },
    otherStatus: {
        backgroundColor: "orange",
        color: "white",
    },
    image: {
        width: 100,
        height: 50,
        objectFit: "contain",
    },
}));

function MTable() {
    const classes = useStyles();
    const [data, setData] = useState([]);
    const [page, setPage] = useState(0);
    const [rowsPerPage, setRowsPerPage] = useState(5);

    useEffect(() => {
        // Set the data from the imported JSON
        setData(jsonData);
    }, []);

    const handleChangePage = (event, newPage) => {
        setPage(newPage);
    };

    const handleChangeRowsPerPage = (event) => {
        setRowsPerPage(+event.target.value);
        setPage(0);
    };

    return (
        <TableContainer component={Paper} className={classes.tableContainer}>
            <Table className={classes.table} aria-label="simple table">
                <TableHead>
                    <TableRow>
                        {data.length > 0 &&
                            Object.keys(data[0]).map((header, index) => (
                                <TableCell key={index} className={classes.tableHeaderCell}>
                                    {header === "Number Plate Image" ? (
                                        <>
                                            <span>{header}</span>
                                            <br />
                                            {/* <span>Vehicle Number</span> */}
                                        </>
                                    ) : (
                                        header
                                    )}
                                </TableCell>
                            ))}
                    </TableRow>
                </TableHead>
                <TableBody>
                    {data.length > 0 &&
                        data
                            .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                            .map((row, index) => (
                                <TableRow key={index}>
                                    {Object.keys(row).map((key, cellIndex) => (
                                        <TableCell key={cellIndex}>
                                            {key === "Number Plate Image" ? (
                                                <img
                                                    src={row[key]}
                                                    alt={`Log ${row.id}`}
                                                    className={classes.image}
                                                />
                                            ) : key === "Status" ? (
                                                <Typography
                                                    className={`${classes.status} ${row[key] === "Entry"
                                                        ? classes.entryStatus
                                                        : row[key] === "Exit"
                                                            ? classes.exitStatus
                                                            : classes.otherStatus
                                                        }`}
                                                >
                                                    {row[key]}
                                                </Typography>
                                            ) : key === "Vehicle Number" ? (
                                                row[key]
                                            ) : (
                                                row[key]
                                            )}
                                        </TableCell>
                                    ))}
                                </TableRow>
                            ))}
                </TableBody>
                <TableFooter>
                    <TableRow>
                        <TablePagination
                            rowsPerPageOptions={[5, 10, 15]}
                            colSpan={data.length > 0 ? Object.keys(data[0]).length : 1}
                            count={data.length}
                            rowsPerPage={rowsPerPage}
                            page={page}
                            onChangePage={handleChangePage}
                            onChangeRowsPerPage={handleChangeRowsPerPage}
                        />
                    </TableRow>
                </TableFooter>
            </Table>
        </TableContainer>
    );
}

export default MTable;
